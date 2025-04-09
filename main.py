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
    
    # Asegurarse de que los géneros sean listas
    if 'generos' in df_filtrado.columns:
        df_filtrado['generos'] = df_filtrado['generos'].apply(
            lambda x: x.split(',') if isinstance(x, str) else []
        )
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
            "/buscar/{query}"
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

@app.get("/buscar/{query}")
async def buscar_peliculas(
    query: str,
    limit: int = Query(5, ge=1, le=20)
):
    try:
        # Verificar que el DataFrame esté cargado
        if df_filtrado is None or df_filtrado.empty:
            logger.error("El DataFrame está vacío o no se ha cargado correctamente")
            return {
                "error": True,
                "mensaje": "No hay datos disponibles para la búsqueda",
                "total_resultados": 0,
                "resultados": []
            }
        
        # Limpiar la consulta
        query = query.lower().strip()
        if not query:
            return {
                "error": True,
                "mensaje": "La consulta no puede estar vacía",
                "total_resultados": 0,
                "resultados": []
            }
        
        # Búsqueda en títulos (más rápida)
        titulos_match = df_filtrado[df_filtrado['titulo'].str.lower().str.contains(query, na=False)]
        
        # Si encontramos suficientes resultados en títulos, no buscamos en otros campos
        if len(titulos_match) >= limit:
            resultados = titulos_match.head(limit)
        else:
            # Búsqueda en sinopsis (solo si es necesario)
            sinopsis_match = df_filtrado[df_filtrado['sinopsis'].str.lower().str.contains(query, na=False)]
            
            # Búsqueda en géneros (solo si es necesario)
            # Convertir géneros a string para evitar problemas con arrays
            generos_match = df_filtrado[df_filtrado['generos'].apply(
                lambda x: any(query in str(g).lower() for g in x) if isinstance(x, list) else False
            )]
            
            # Combinar resultados
            resultados = pd.concat([titulos_match, sinopsis_match, generos_match]).drop_duplicates()
        
        # Ordenar y limitar resultados
        resultados = resultados.sort_values('puntuacion', ascending=False).head(limit)
        
        # Convertir a diccionario de manera eficiente y segura
        resultados_dict = []
        for _, row in resultados.iterrows():
            # Convertir géneros a lista de strings para evitar problemas con numpy arrays
            generos = row['generos']
            if isinstance(generos, np.ndarray):
                generos = generos.tolist()
            elif not isinstance(generos, list):
                generos = [str(generos)]
            
            # Asegurarse de que todos los valores sean serializables
            pelicula = {
                "titulo": str(row['titulo']),
                "sinopsis": str(row['sinopsis']),
                "puntuacion": float(row['puntuacion']),
                "generos": [str(g) for g in generos]
            }
            resultados_dict.append(pelicula)
        
        return {
            "error": False,
            "mensaje": "Búsqueda completada exitosamente",
            "total_resultados": len(resultados_dict),
            "resultados": resultados_dict
        }
    except Exception as e:
        logger.error(f"Error en buscar_peliculas: {str(e)}")
        # Devolver un error 500 con mensaje claro en lugar de lanzar excepción
        return {
            "error": True,
            "mensaje": "Error al procesar la búsqueda",
            "detalle": str(e),
            "total_resultados": 0,
            "resultados": []
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 