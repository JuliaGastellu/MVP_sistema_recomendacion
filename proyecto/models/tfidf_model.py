"""
Modelo de Recomendación basado en TF-IDF y Similitud del Coseno

Este módulo implementa un sistema de recomendación que utiliza TF-IDF y similitud del coseno
para encontrar películas similares basadas en sus características textuales.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
import re
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Descargar recursos necesarios de NLTK
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    spanish_stopwords = stopwords.words('spanish')
    logger.info("Recursos NLTK descargados correctamente")
except Exception as e:
    logger.error(f"Error al descargar recursos NLTK: {str(e)}")
    spanish_stopwords = []

class TFIDFRecommender:
    """
    Clase para el sistema de recomendación basado en TF-IDF y similitud del coseno.
    """
    
    def __init__(self, data_path=None):
        """
        Inicializa el recomendador con la ruta al archivo de datos.
        
        Args:
            data_path (str): Ruta al archivo de datos en formato Parquet.
        """
        self.data_path = data_path
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.tfidf = None
        
        if data_path:
            self.load_data()
    
    def load_data(self):
        """
        Carga los datos desde el archivo Parquet y realiza el preprocesamiento inicial.
        """
        try:
            logger.info(f"Cargando datos desde {self.data_path}")
            self.df = pd.read_parquet(self.data_path)
            self._preprocess_data()
            logger.info("Datos cargados y preprocesados correctamente")
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def _clean_text(self, text):
        """
        Limpia y normaliza el texto.
        
        Args:
            text (str): Texto a limpiar.
            
        Returns:
            str: Texto limpio y normalizado.
        """
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Eliminar espacios extra
        text = ' '.join(text.split())
        
        return text
    
    def _process_reviews(self, reviews):
        """
        Procesa las reseñas, que pueden ser una lista o un string.
        
        Args:
            reviews: Reseñas a procesar (puede ser lista o string).
            
        Returns:
            str: Texto procesado de las reseñas.
        """
        if pd.isna(reviews):
            return ""
        
        if isinstance(reviews, list):
            return ' '.join([self._clean_text(review) for review in reviews])
        else:
            return self._clean_text(str(reviews))
    
    def _preprocess_data(self):
        """
        Preprocesa los datos para la vectorización.
        """
        # Limpiar y normalizar los datos
        self.df['titulo_clean'] = self.df['titulo'].apply(self._clean_text)
        self.df['sinopsis_clean'] = self.df['sinopsis'].apply(self._clean_text)
        self.df['reseñas_clean'] = self.df['reseñas'].apply(self._process_reviews)
        self.df['generos_clean'] = self.df['generos'].apply(lambda x: self._clean_text(str(x)))
        
        # Crear una columna 'caracteristicas' combinando géneros, reseñas y sinopsis con pesos diferentes
        # Géneros tienen más peso (3x), sinopsis (2x) y reseñas (1x)
        self.df['caracteristicas'] = self.df['generos_clean'].apply(lambda x: ' '.join([x] * 3)) + ' ' + \
                                    self.df['sinopsis_clean'].apply(lambda x: ' '.join([x] * 2)) + ' ' + \
                                    self.df['reseñas_clean']
    
    def fit(self):
        """
        Entrena el modelo TF-IDF y calcula la matriz de similitud.
        """
        try:
            logger.info("Entrenando modelo TF-IDF")
            
            # Configurar TfidfVectorizer con parámetros optimizados
            self.tfidf = TfidfVectorizer(
                stop_words=spanish_stopwords,
                max_features=10000,  # Limitar el número de características para mejorar el rendimiento
                ngram_range=(1, 2),  # Incluir pares de palabras para capturar mejor el contexto
                min_df=2,  # Ignorar términos que aparecen en menos de 2 documentos
                max_df=0.95  # Ignorar términos que aparecen en más del 95% de los documentos
            )
            
            # Aplicar TF-IDF a las características
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['caracteristicas'])
            
            # Calcular la similitud del coseno entre las películas
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            logger.info("Modelo TF-IDF entrenado correctamente")
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {str(e)}")
            raise
    
    def recommend_by_title(self, title, top_n=10):
        """
        Recomienda películas similares basadas en el título.
        
        Args:
            title (str): Título de la película.
            top_n (int): Número de recomendaciones a devolver.
            
        Returns:
            pd.DataFrame: DataFrame con las películas recomendadas.
        """
        try:
            # Buscar el título de la película (ignorando mayúsculas/minúsculas)
            title = title.lower()
            
            # Verificar si el título de la película existe en el DataFrame
            matching_movies = self.df[self.df['titulo_clean'].str.contains(title, na=False)]
            
            if matching_movies.empty:
                # Buscar títulos similares
                similar_titles = []
                for idx, row in self.df.iterrows():
                    if title in row['titulo_clean'] or row['titulo_clean'] in title:
                        similar_titles.append(row['titulo'])
                
                if similar_titles:
                    logger.info(f"Título no encontrado exactamente. Sugerencias: {similar_titles[:5]}")
                    return {
                        'error': True,
                        'message': f'La película "{title}" no se encuentra exactamente. ¿Quizás quisiste decir: {", ".join(similar_titles[:5])}?',
                        'suggestions': similar_titles[:5]
                    }
                else:
                    logger.info(f"Título no encontrado: {title}")
                    return {
                        'error': True,
                        'message': f'La película "{title}" no se encuentra en la base de datos.'
                    }
            
            # Si hay múltiples coincidencias, usar la primera
            idx = matching_movies.index[0]
            
            # Obtener las puntuaciones de similitud con todas las películas
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Ordenar las películas basándome en las similitudes
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            # Obtener los índices de las películas más similares
            movie_indices = [i[0] for i in sim_scores]
            
            # Obtener las películas recomendadas con sus puntuaciones de similitud
            recommendations = self.df.iloc[movie_indices][["titulo", "sinopsis", "puntuacion", "generos"]]
            recommendations['similitud'] = [i[1] for i in sim_scores]
            
            # Normalizar la puntuación de similitud para que esté entre 0 y 1
            scaler = MinMaxScaler()
            recommendations['similitud'] = scaler.fit_transform(recommendations[['similitud']])
            
            logger.info(f"Recomendaciones generadas para: {title}")
            return {
                'error': False,
                'recommendations': recommendations.to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {str(e)}")
            return {
                'error': True,
                'message': f"Error al generar recomendaciones: {str(e)}"
            }
    
    def recommend_by_genre(self, genre, top_n=10):
        """
        Recomienda películas basadas en el género.
        
        Args:
            genre (str): Género de la película.
            top_n (int): Número de recomendaciones a devolver.
            
        Returns:
            pd.DataFrame: DataFrame con las películas recomendadas.
        """
        try:
            # Limpiar el género de entrada
            genre = self._clean_text(genre)
            
            # Filtrar películas que contienen el género
            matching_movies = self.df[self.df['generos_clean'].str.contains(genre, na=False)]
            
            if matching_movies.empty:
                logger.info(f"Género no encontrado: {genre}")
                return {
                    'error': True,
                    'message': f'No se encontraron películas con el género "{genre}".'
                }
            
            # Ordenar por puntuación y devolver las mejores
            recommendations = matching_movies.sort_values(by='puntuacion', ascending=False).head(top_n)[["titulo", "sinopsis", "puntuacion", "generos"]]
            
            logger.info(f"Recomendaciones por género generadas para: {genre}")
            return {
                'error': False,
                'recommendations': recommendations.to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error al generar recomendaciones por género: {str(e)}")
            return {
                'error': True,
                'message': f"Error al generar recomendaciones por género: {str(e)}"
            }
    
    def search_movies(self, query, limit=10):
        """
        Busca películas por título o descripción.
        
        Args:
            query (str): Consulta de búsqueda.
            limit (int): Número máximo de resultados.
            
        Returns:
            dict: Resultados de la búsqueda.
        """
        try:
            query = self._clean_text(query)
            
            # Buscar por título
            title_matches = self.df[self.df['titulo_clean'].str.contains(query, case=False, na=False)]
            
            # Buscar por sinopsis
            synopsis_matches = self.df[self.df['sinopsis_clean'].str.contains(query, case=False, na=False)]
            
            # Combinar resultados
            results = pd.concat([title_matches, synopsis_matches]).drop_duplicates().head(limit)
            
            if results.empty:
                logger.info(f"No se encontraron resultados para: {query}")
                return {
                    'error': False,
                    'message': "No se encontraron películas",
                    'results': []
                }
            
            logger.info(f"Búsqueda completada para: {query}")
            return {
                'error': False,
                'message': f"Se encontraron {len(results)} películas",
                'results': results[["titulo", "sinopsis", "puntuacion"]].to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error en búsqueda: {str(e)}")
            return {
                'error': True,
                'message': f"Error en búsqueda: {str(e)}"
            }


# Función para cargar el modelo desde la ruta del proyecto
def load_tfidf_model():
    """
    Carga el modelo TF-IDF desde la ruta del proyecto.
    
    Returns:
        TFIDFRecommender: Modelo de recomendación cargado.
    """
    try:
        # Obtener la ruta base del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'movies_filtrado.parquet')
        
        # Crear y cargar el modelo
        model = TFIDFRecommender(data_path)
        model.fit()
        
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo TF-IDF: {str(e)}")
        raise


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Cargar el modelo
        model = load_tfidf_model()
        
        # Probar recomendaciones
        title = "Matrix"
        recommendations = model.recommend_by_title(title)
        
        if not recommendations['error']:
            print(f"Recomendaciones para '{title}':")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"{i}. {rec['titulo']} (Similitud: {rec['similitud'][0]:.2f})")
        else:
            print(recommendations['message'])
        
        # Probar búsqueda
        query = "acción"
        search_results = model.search_movies(query)
        
        if not search_results['error']:
            print(f"\nResultados de búsqueda para '{query}':")
            for i, result in enumerate(search_results['results'], 1):
                print(f"{i}. {result['titulo']}")
        else:
            print(search_results['message'])
    except Exception as e:
        print(f"Error: {str(e)}") 