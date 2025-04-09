import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    """Clase para manejar las recomendaciones de películas."""
    
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializar el recomendador de películas.
        
        Args:
            data_path: Ruta al archivo de datos
            model_name: Nombre del modelo de Sentence Transformers
        """
        try:
            # Cargar datos
            self.df = pd.read_parquet(data_path)
            self.df['reseñas'] = self.df['reseñas'].fillna('')
            
            # Configurar TF-IDF
            nltk.download('stopwords')
            self.stopwords_es = stopwords.words('spanish')
            self.tfidf = TfidfVectorizer(stop_words=self.stopwords_es)
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['reseñas'])
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            # Configurar índices
            self.indices = pd.Series(self.df.index, index=self.df['titulo'].str.lower()).drop_duplicates()
            
            # Configurar modelo de embeddings
            self.model = SentenceTransformer(model_name)
            self.df["generos"] = self.df["generos"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
            self.df["vector"] = self.df["generos"].apply(lambda x: self._get_movie_vector(x))
            
            logger.info("MovieRecommender inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar MovieRecommender: {str(e)}")
            raise
    
    def _get_movie_vector(self, genres: List[str]) -> np.ndarray:
        """Obtener vector de embeddings para géneros."""
        genres_text = " ".join(genres)
        return self.model.encode([genres_text])[0]
    
    def get_recommendations_by_reviews(self, titulo: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones basadas en reseñas.
        
        Args:
            titulo: Título de la película
            top_n: Número de recomendaciones a devolver
            
        Returns:
            Lista de diccionarios con recomendaciones
        """
        try:
            titulo = titulo.strip().lower()
            if titulo not in self.indices:
                raise ValueError("Película no encontrada")
            
            idx = self.indices[titulo]
            sim_scores = sorted(list(enumerate(self.cosine_sim[idx])), 
                              key=lambda x: x[1], reverse=True)[1:top_n+1]
            movie_indices = [i[0] for i in sim_scores]
            
            recomendaciones = self.df.iloc[movie_indices][["titulo", "sinopsis", "puntuacion"]]
            return recomendaciones.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error en get_recommendations_by_reviews: {str(e)}")
            raise
    
    def get_recommendations_by_genres(self, titulo: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Obtener recomendaciones basadas en géneros.
        
        Args:
            titulo: Título de la película
            top_n: Número de recomendaciones a devolver
            
        Returns:
            Lista de diccionarios con recomendaciones
        """
        try:
            movie_row = self.df[self.df['titulo'].str.contains(titulo, case=False, na=False)]
            if movie_row.empty:
                raise ValueError("Película no encontrada")
            
            movie_vector = movie_row.iloc[0]["vector"].reshape(1, -1)
            similarities = cosine_similarity(movie_vector, np.stack(self.df["vector"].values))
            self.df["similarity"] = similarities[0]
            
            recomendaciones = (self.df.sort_values(by="similarity", ascending=False)
                            .head(top_n)[["titulo", "generos", "sinopsis", "puntuacion"]])
            
            recomendaciones["generos"] = recomendaciones["generos"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else "")
            
            return recomendaciones.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error en get_recommendations_by_genres: {str(e)}")
            raise 