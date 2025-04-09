from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from app.core.config import get_settings
from app.models.recommendation import MovieRecommender
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()

# Obtener configuración
settings = get_settings()

# Inicializar recomendador
try:
    recommender = MovieRecommender(settings.DATA_PATH, settings.MODEL_NAME)
    logger.info("Recomendador inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el recomendador: {str(e)}")
    raise

@router.get("/recomendacion/{titulo}", response_model=List[Dict[str, Any]])
async def get_recommendations_by_reviews(
    titulo: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Obtener recomendaciones basadas en reseñas.
    
    Args:
        titulo: Título de la película
        top_n: Número de recomendaciones a devolver
        
    Returns:
        Lista de recomendaciones
    """
    try:
        return recommender.get_recommendations_by_reviews(titulo, top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en get_recommendations_by_reviews: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/recomendacion_genero/{titulo}", response_model=List[Dict[str, Any]])
async def get_recommendations_by_genres(
    titulo: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Obtener recomendaciones basadas en géneros.
    
    Args:
        titulo: Título de la película
        top_n: Número de recomendaciones a devolver
        
    Returns:
        Lista de recomendaciones
    """
    try:
        return recommender.get_recommendations_by_genres(titulo, top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error en get_recommendations_by_genres: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor") 