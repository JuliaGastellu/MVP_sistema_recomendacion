from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import re
import difflib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.error(f"Error al descargar recursos NLTK: {str(e)}")

# Descargar stopwords en español
try:
    nltk.download('stopwords')
    stopwords_es = stopwords.words('spanish')
except Exception as e:
    logger.error(f"Error al descargar stopwords: {str(e)}")
    stopwords_es = []

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'proyecto', 'data', 'movies_filtrado.parquet')

# Verificar que el archivo existe
if not os.path.exists(data_path):
    logger.error(f"Archivo de datos no encontrado en: {data_path}")
    raise FileNotFoundError(f"Archivo de datos no encontrado en: {data_path}")

# Cargar datos una sola vez al inicio
try:
    df_filtrado = pd.read_parquet(data_path)
    logger.info(f"Datos cargados exitosamente. Total de películas: {len(df_filtrado)}")
    
    # Asegurarse de que los géneros sean listas de strings
    if 'generos' in df_filtrado.columns:
        # Convertir géneros a listas de strings
        df_filtrado['generos'] = df_filtrado['generos'].apply(
            lambda x: [str(g).strip() for g in x.split(',')] if isinstance(x, str) else []
        )
        
        # Verificar que no haya arrays de numpy
        for idx, row in df_filtrado.iterrows():
            if isinstance(row['generos'], np.ndarray):
                df_filtrado.at[idx, 'generos'] = [str(g) for g in row['generos']]
            elif not isinstance(row['generos'], list):
                df_filtrado.at[idx, 'generos'] = [str(row['generos'])]
except Exception as e:
    logger.error(f"Error al cargar los datos: {str(e)}")
    raise

# Configurar el modelo de recomendación
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Modelo SentenceTransformer cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

# Cache para recomendaciones
recommendation_cache = {}
CACHE_TTL = 3600  # 1 hora en segundos

# Función para limpiar títulos
def clean_title(title):
    try:
        return title.lower().strip()
    except:
        return ""

# Función para encontrar títulos similares
def find_similar_titles(query, df, threshold=0.8):
    try:
        query = clean_title(query)
        if not query:
            return None
        
        # Buscar coincidencia exacta primero
        exact_match = df[df['titulo'].str.lower() == query]
        if not exact_match.empty:
            return exact_match.iloc[0]['titulo']
        
        # Si no hay coincidencia exacta, buscar similitud
        titles = df['titulo'].str.lower().tolist()
        similarities = [difflib.SequenceMatcher(None, query, title).ratio() for title in titles]
        max_similarity = max(similarities)
        
        if max_similarity >= threshold:
            return df.iloc[similarities.index(max_similarity)]['titulo']
        return None
    except Exception as e:
        logger.error(f"Error en find_similar_titles: {str(e)}")
        return None

# Función para generar recomendaciones con caché
@lru_cache(maxsize=100)
def get_recommendations(title, limit=5):
    try:
        if title in recommendation_cache:
            return recommendation_cache[title]
        
        similar_title = find_similar_titles(title, df_filtrado)
        if not similar_title:
            return None
        
        movie_data = df_filtrado[df_filtrado['titulo'] == similar_title].iloc[0]
        genres = movie_data['generos']
        
        # Filtrar por géneros y calcular similitud
        genre_matches = df_filtrado[df_filtrado['generos'].apply(lambda x: any(g in genres for g in x))]
        if len(genre_matches) > 1:
            genre_matches = genre_matches[genre_matches['titulo'] != similar_title]
        
        recommendations = genre_matches.head(limit).to_dict('records')
        recommendation_cache[title] = recommendations
        return recommendations
    except Exception as e:
        logger.error(f"Error en get_recommendations: {str(e)}")
        return None

# Función para manejar excepciones
async def handle_exception(func):
    try:
        return await func()
    except Exception as e:
        logger.error(f"Error en {func.__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

# Modelos Pydantic
class MovieRecommendation(PydanticBaseModel):
    titulo: str
    sinopsis: str
    puntuacion: float
    generos: List[str]

class RecommendationResponse(PydanticBaseModel):
    error: bool
    mensaje: str
    peliculas: List[MovieRecommendation]

class ErrorResponse(PydanticBaseModel):
    error: bool
    mensaje: str
    detalle: Optional[str] = None

# Aplicación FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="API para recomendar películas basada en similitud de reseñas y géneros",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=True,
            mensaje="Error interno del servidor. Por favor, inténtelo de nuevo más tarde.",
            detalle=str(exc)
        ).dict()
    )

@app.get("/")
async def root():
    return {
        "mensaje": "Bienvenido al Sistema de Recomendación de Películas",
        "endpoints_disponibles": [
            "/recomendacion/{titulo}",
            "/recomendacion_genero/{titulo}",
            "/buscar/{query}",
            "/peliculas/filtradas"
        ]
    }

@app.get("/recomendacion/{titulo}", response_model=RecommendationResponse)
async def recomendar_peliculas(titulo: str):
    try:
        recommendations = get_recommendations(titulo)
        if not recommendations:
            return RecommendationResponse(
                error=True,
                mensaje=f"No se encontraron recomendaciones para '{titulo}'",
                peliculas=[]
            )
        
        return RecommendationResponse(
            error=False,
            mensaje=f"Recomendaciones para '{titulo}'",
            peliculas=recommendations
        )
    except Exception as e:
        logger.error(f"Error en recomendar_peliculas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar recomendaciones: {str(e)}"
        )

@app.get("/recomendacion_genero/{titulo}", response_model=RecommendationResponse)
async def recomendar_por_genero(titulo: str):
    try:
        similar_title = find_similar_titles(titulo, df_filtrado)
        if not similar_title:
            return RecommendationResponse(
                error=True,
                mensaje=f"No se encontró la película '{titulo}'",
                peliculas=[]
            )
        
        movie_data = df_filtrado[df_filtrado['titulo'] == similar_title].iloc[0]
        genres = movie_data['generos']
        
        recommendations = df_filtrado[
            df_filtrado['generos'].apply(lambda x: any(g in genres for g in x)) &
            (df_filtrado['titulo'] != similar_title)
        ].head(5).to_dict('records')
        
        return RecommendationResponse(
            error=False,
            mensaje=f"Recomendaciones por género para '{similar_title}'",
            peliculas=recommendations
        )
    except Exception as e:
        logger.error(f"Error en recomendar_por_genero: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar recomendaciones por género: {str(e)}"
        )

@app.get("/peliculas/filtradas")
async def filtrar_peliculas(
    genero: str = Query(None, description="Género de la película"),
    puntuacion_min: float = Query(0.0, ge=0.0, le=10.0, description="Puntuación mínima"),
    puntuacion_max: float = Query(10.0, ge=0.0, le=10.0, description="Puntuación máxima"),
    limit: int = Query(10, ge=1, le=50, description="Número máximo de resultados")
):
    """
    Filtra películas por género y rango de puntuación.
    """
    try:
        # Verificar que el DataFrame esté cargado
        if df_filtrado is None or df_filtrado.empty:
            logger.error("El DataFrame está vacío o no se ha cargado correctamente")
            return {
                "error": True,
                "mensaje": "No hay datos disponibles para el filtrado",
                "total_resultados": 0,
                "resultados": []
            }
        
        # Filtrar por puntuación
        df_filtrado_puntuacion = df_filtrado[
            (df_filtrado['puntuacion'] >= puntuacion_min) & 
            (df_filtrado['puntuacion'] <= puntuacion_max)
        ]
        
        # Si se especifica un género, filtrar por él
        if genero:
            genero = genero.lower().strip()
            df_filtrado_final = df_filtrado_puntuacion[
                df_filtrado_puntuacion['generos'].apply(
                    lambda x: any(g.lower().strip() == genero for g in x)
                )
            ]
        else:
            df_filtrado_final = df_filtrado_puntuacion
        
        # Ordenar por puntuación y limitar resultados
        resultados = df_filtrado_final.sort_values('puntuacion', ascending=False).head(limit)
        
        # Convertir a diccionario
        resultados_dict = []
        for _, row in resultados.iterrows():
            pelicula = {
                "titulo": str(row['titulo']),
                "sinopsis": str(row['sinopsis']),
                "puntuacion": float(row['puntuacion']),
                "generos": [str(g) for g in row['generos']]
            }
            resultados_dict.append(pelicula)
        
        return {
            "error": False,
            "mensaje": "Filtrado completado exitosamente",
            "total_resultados": len(resultados_dict),
            "resultados": resultados_dict
        }
    except Exception as e:
        logger.error(f"Error en filtrar_peliculas: {str(e)}")
        return {
            "error": True,
            "mensaje": "Error en el filtrado",
            "detalle": str(e),
            "total_resultados": 0,
            "resultados": []
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 