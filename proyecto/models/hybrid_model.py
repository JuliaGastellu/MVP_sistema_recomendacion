"""
Modelo de Recomendación Híbrido

Este módulo implementa un sistema de recomendación híbrido que combina
TF-IDF y Sentence Transformers para obtener recomendaciones más precisas.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import os
import logging
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import gc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Descargar recursos necesarios de NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    logger.info("Recursos NLTK descargados correctamente en hybrid_model")
except Exception as e:
    logger.error(f"Error al descargar recursos NLTK en hybrid_model: {str(e)}")
    raise

class HybridRecommender:
    """
    Clase para el sistema de recomendación híbrido.
    """
    
    def __init__(self, data_path=None, model_name='all-MiniLM-L6-v2', tfidf_weight=0.7):
        """
        Inicializa el recomendador híbrido.
        
        Args:
            data_path (str): Ruta al archivo de datos en formato Parquet.
            model_name (str): Nombre del modelo de Sentence Transformers.
            tfidf_weight (float): Peso para las similitudes de TF-IDF (0-1).
        """
        self.data_path = data_path
        self.model_name = model_name
        self.tfidf_weight = tfidf_weight
        self.st_weight = 1 - tfidf_weight
        
        self.df = None
        self.st_model = None
        self.tfidf = None
        self.st_vectors = None
        self.tfidf_matrix = None
        self.stemmer = SnowballStemmer('spanish')
        
        # Forzar liberación de memoria
        gc.collect()
        
        if data_path:
            self.load_data()
            self.load_models()
    
    def load_data(self):
        """
        Carga y preprocesa los datos.
        """
        try:
            logger.info(f"Cargando datos desde {self.data_path}")
            # Cargar solo las columnas necesarias para reducir el uso de memoria
            self.df = pd.read_parquet(
                self.data_path,
                columns=['titulo', 'sinopsis', 'generos', 'anio_estreno', 'puntuacion']
            )
            self._preprocess_data()
            logger.info("Datos cargados y preprocesados correctamente")
            # Forzar liberación de memoria
            gc.collect()
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def load_models(self):
        """
        Carga los modelos de Sentence Transformers y TF-IDF.
        """
        try:
            # Cargar Sentence Transformer con configuración optimizada
            logger.info(f"Cargando modelo {self.model_name}")
            self.st_model = SentenceTransformer(
                self.model_name,
                device='cpu',
                cache_folder='/opt/render/project/src/.cache/huggingface'
            )
            
            # Asegurarse de que stopwords esté disponible
            try:
                spanish_stopwords = stopwords.words('spanish')
            except LookupError:
                logger.info("Descargando stopwords para español...")
                nltk.download('stopwords', quiet=True)
                spanish_stopwords = stopwords.words('spanish')
            
            # Configurar TF-IDF con parámetros optimizados
            self.tfidf = TfidfVectorizer(
                stop_words=spanish_stopwords,
                max_features=500,  # Reducido de 1000
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                dtype=np.float32  # Usar float32 en lugar de float64
            )
            
            logger.info("Modelos cargados correctamente")
            # Forzar liberación de memoria
            gc.collect()
        except Exception as e:
            logger.error(f"Error al cargar modelos: {str(e)}")
            raise
    
    def _clean_text(self, text):
        """
        Limpia y normaliza el texto.
        """
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def _stem_text(self, text):
        """
        Aplica stemming al texto.
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def _process_reviews(self, reviews):
        """
        Procesa las reseñas.
        """
        if pd.isna(reviews):
            return ""
        
        if isinstance(reviews, list):
            return ' '.join([self._clean_text(review) for review in reviews])
        else:
            return self._clean_text(str(reviews))
    
    def _preprocess_data(self):
        """
        Preprocesa los datos para ambos modelos.
        """
        # Limpiar y normalizar datos
        self.df['titulo_clean'] = self.df['titulo'].apply(self._clean_text)
        self.df['sinopsis_clean'] = self.df['sinopsis'].apply(self._clean_text)
        self.df['generos_clean'] = self.df['generos'].apply(lambda x: self._clean_text(str(x)))
        
        # Aplicar stemming para TF-IDF
        self.df['texto_tfidf'] = self.df.apply(
            lambda row: f"{self._stem_text(row['titulo_clean'])} {self._stem_text(row['generos_clean'])} {self._stem_text(row['sinopsis_clean'])}",
            axis=1
        )
        
        # Texto para Sentence Transformer (sin stemming)
        self.df['texto_st'] = self.df.apply(
            lambda row: f"{row['titulo_clean']} {row['generos_clean']} {row['sinopsis_clean']}",
            axis=1
        )
    
    def fit(self):
        """
        Entrena los modelos con los datos cargados.
        """
        try:
            logger.info("Entrenando modelos híbridos")
            
            # Entrenar TF-IDF
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['sinopsis_clean'])
            
            # Generar embeddings con Sentence Transformer en lotes más pequeños
            batch_size = 4  # Reducido de 8
            texts = self.df['sinopsis_clean'].tolist()
            self.st_vectors = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.st_model.encode(batch, show_progress_bar=False)
                self.st_vectors.append(batch_embeddings)
                # Forzar liberación de memoria después de cada lote
                gc.collect()
            
            self.st_vectors = np.vstack(self.st_vectors)
            
            logger.info("Modelos entrenados correctamente")
            # Forzar liberación de memoria
            gc.collect()
        except Exception as e:
            logger.error(f"Error al entrenar modelos: {str(e)}")
            raise
    
    def _get_hybrid_similarities(self, idx):
        """
        Calcula las similitudes híbridas para un índice dado.
        """
        # Similitud TF-IDF
        tfidf_sim = cosine_similarity(
            self.tfidf_matrix[idx:idx+1],
            self.tfidf_matrix
        )[0]
        
        # Similitud Sentence Transformer
        st_sim = cosine_similarity(
            self.st_vectors[idx:idx+1],
            self.st_vectors
        )[0]
        
        # Combinar similitudes
        hybrid_sim = (self.tfidf_weight * tfidf_sim) + (self.st_weight * st_sim)
        
        return hybrid_sim
    
    def recommend_by_title(self, title, top_n=10):
        """
        Recomienda películas similares basadas en el título.
        """
        try:
            title = title.lower()
            matching_movies = self.df[self.df['titulo_clean'].str.contains(title, na=False)]
            
            if matching_movies.empty:
                similar_titles = []
                for idx, row in self.df.iterrows():
                    if title in row['titulo_clean'] or row['titulo_clean'] in title:
                        similar_titles.append(row['titulo'])
                
                if similar_titles:
                    return {
                        'error': True,
                        'message': f'La película "{title}" no se encuentra exactamente. ¿Quizás quisiste decir: {", ".join(similar_titles[:5])}?',
                        'suggestions': similar_titles[:5]
                    }
                else:
                    return {
                        'error': True,
                        'message': f'La película "{title}" no se encuentra en la base de datos.'
                    }
            
            idx = matching_movies.index[0]
            similarities = self._get_hybrid_similarities(idx)
            
            similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
            
            recommendations = self.df.iloc[similar_indices][["titulo", "sinopsis", "puntuacion", "generos", "anio_estreno"]]
            recommendations['similitud'] = similarities[similar_indices]
            
            scaler = MinMaxScaler()
            recommendations['similitud'] = scaler.fit_transform(recommendations[['similitud']])
            
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
        """
        try:
            genre = self._clean_text(genre)
            matching_movies = self.df[self.df['generos_clean'].str.contains(genre, na=False)]
            
            if matching_movies.empty:
                return {
                    'error': True,
                    'message': f'No se encontraron películas con el género "{genre}".'
                }
            
            # Calcular similitudes híbridas para todas las películas del género
            similarities = np.zeros(len(self.df))
            for idx in matching_movies.index:
                similarities += self._get_hybrid_similarities(idx)
            
            # Normalizar similitudes
            similarities = similarities / len(matching_movies)
            
            # Obtener las películas más similares
            similar_indices = np.argsort(similarities)[::-1][:top_n]
            
            recommendations = self.df.iloc[similar_indices][["titulo", "sinopsis", "puntuacion", "generos", "anio_estreno"]]
            recommendations['similitud'] = similarities[similar_indices]
            
            scaler = MinMaxScaler()
            recommendations['similitud'] = scaler.fit_transform(recommendations[['similitud']])
            
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
    
    def recommend_by_year(self, year, top_n=10):
        """
        Recomienda películas basadas en el año de lanzamiento.
        """
        try:
            matching_movies = self.df[self.df['anio_estreno'] == year]
            
            if matching_movies.empty:
                return {
                    'error': True,
                    'message': f'No se encontraron películas del año {year}.'
                }
            
            # Ordenar por puntuación y devolver las mejores
            recommendations = matching_movies.sort_values(by='puntuacion', ascending=False).head(top_n)[["titulo", "sinopsis", "puntuacion", "generos", "anio_estreno"]]
            
            return {
                'error': False,
                'recommendations': recommendations.to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error al generar recomendaciones por año: {str(e)}")
            return {
                'error': True,
                'message': f"Error al generar recomendaciones por año: {str(e)}"
            }
    
    def search_movies(self, query, limit=10):
        """
        Busca películas por título o descripción.
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
                return {
                    'error': False,
                    'message': "No se encontraron películas",
                    'results': []
                }
            
            return {
                'error': False,
                'message': f"Se encontraron {len(results)} películas",
                'results': results[["titulo", "sinopsis", "puntuacion", "generos", "anio_estreno"]].to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error en búsqueda: {str(e)}")
            return {
                'error': True,
                'message': f"Error en búsqueda: {str(e)}"
            }


# Función para cargar el modelo desde la ruta del proyecto
def load_hybrid_model():
    """
    Carga el modelo híbrido desde la ruta del proyecto.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'movies_filtrado.parquet')
        
        model = HybridRecommender(data_path)
        model.fit()
        
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo híbrido: {str(e)}")
        raise


# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Cargar el modelo
        model = load_hybrid_model()
        
        # Probar recomendaciones por título
        title = "Matrix"
        recommendations = model.recommend_by_title(title)
        
        if not recommendations['error']:
            print(f"Recomendaciones para '{title}':")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                print(f"{i}. {rec['titulo']} (Similitud: {rec['similitud'][0]:.2f})")
        else:
            print(recommendations['message'])
        
        # Probar recomendaciones por género
        genre = "acción"
        genre_recommendations = model.recommend_by_genre(genre)
        
        if not genre_recommendations['error']:
            print(f"\nRecomendaciones por género '{genre}':")
            for i, rec in enumerate(genre_recommendations['recommendations'], 1):
                print(f"{i}. {rec['titulo']} (Similitud: {rec['similitud'][0]:.2f})")
        else:
            print(genre_recommendations['message'])
        
        # Probar recomendaciones por año
        year = 2020
        year_recommendations = model.recommend_by_year(year)
        
        if not year_recommendations['error']:
            print(f"\nRecomendaciones del año {year}:")
            for i, rec in enumerate(year_recommendations['recommendations'], 1):
                print(f"{i}. {rec['titulo']} (Puntuación: {rec['puntuacion']})")
        else:
            print(year_recommendations['message'])
    except Exception as e:
        print(f"Error: {str(e)}") 