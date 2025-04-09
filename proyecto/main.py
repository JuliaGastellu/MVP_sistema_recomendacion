"""
API de Sistema de Recomendación de Películas

Este módulo implementa una API REST usando FastAPI para el sistema de recomendación
de películas basado en modelos híbridos.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from models.hybrid_model import load_hybrid_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear la aplicación FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="API para recomendar películas basada en diferentes criterios",
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

# Cargar el modelo al iniciar la aplicación
try:
    model = load_hybrid_model()
    logger.info("Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

class RecommendationResponse(BaseModel):
    """Modelo para la respuesta de recomendaciones."""
    error: bool
    message: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None

@app.get("/")
async def root():
    """Endpoint raíz que devuelve información básica sobre la API."""
    return {
        "message": "Bienvenido al Sistema de Recomendación de Películas",
        "version": "1.0.0",
        "endpoints": [
            "/recommend/title/{title}",
            "/recommend/genre/{genre}",
            "/recommend/year/{year}",
            "/search/{query}"
        ]
    }

@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(title: str, top_n: int = 10):
    """
    Obtiene recomendaciones de películas basadas en el título.
    
    Args:
        title (str): Título de la película.
        top_n (int): Número de recomendaciones a devolver (default: 10).
    
    Returns:
        RecommendationResponse: Recomendaciones de películas similares.
    """
    try:
        recommendations = model.recommend_by_title(title, top_n)
        return recommendations
    except Exception as e:
        logger.error(f"Error en recommend_by_title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/genre/{genre}", response_model=RecommendationResponse)
async def recommend_by_genre(genre: str, top_n: int = 10):
    """
    Obtiene recomendaciones de películas basadas en el género.
    
    Args:
        genre (str): Género de la película.
        top_n (int): Número de recomendaciones a devolver (default: 10).
    
    Returns:
        RecommendationResponse: Recomendaciones de películas del género.
    """
    try:
        recommendations = model.recommend_by_genre(genre, top_n)
        return recommendations
    except Exception as e:
        logger.error(f"Error en recommend_by_genre: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/year/{year}", response_model=RecommendationResponse)
async def recommend_by_year(year: int, top_n: int = 10):
    """
    Obtiene recomendaciones de películas basadas en el año.
    
    Args:
        year (int): Año de lanzamiento.
        top_n (int): Número de recomendaciones a devolver (default: 10).
    
    Returns:
        RecommendationResponse: Recomendaciones de películas del año.
    """
    try:
        recommendations = model.recommend_by_year(year, top_n)
        return recommendations
    except Exception as e:
        logger.error(f"Error en recommend_by_year: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{query}", response_model=RecommendationResponse)
async def search_movies(query: str, limit: int = 10):
    """
    Busca películas por título o descripción.
    
    Args:
        query (str): Término de búsqueda.
        limit (int): Número máximo de resultados (default: 10).
    
    Returns:
        RecommendationResponse: Resultados de la búsqueda.
    """
    try:
        results = model.search_movies(query, limit)
        return results
    except Exception as e:
        logger.error(f"Error en search_movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 