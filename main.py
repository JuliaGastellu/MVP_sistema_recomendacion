from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic.v1 import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import sys
import os
import nltk
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
    logger.info("Recursos NLTK descargados correctamente")
except Exception as e:
    logger.error(f"Error al descargar recursos NLTK: {str(e)}")
    raise

# Agregar el directorio proyecto al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'proyecto'))
from models.hybrid_model import load_hybrid_model

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="API para recomendar películas basada en diferentes criterios",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de respuesta
class RecommendationResponse(BaseModel):
    """Modelo para la respuesta de recomendaciones."""
    error: bool
    message: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None

# Variable global para el modelo
model = None

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    global model
    try:
        logger.info("Iniciando carga del modelo híbrido...")
        # Forzar liberación de memoria antes de cargar el modelo
        gc.collect()
        model = load_hybrid_model()
        # Forzar liberación de memoria después de cargar el modelo
        gc.collect()
        logger.info("Modelo híbrido cargado correctamente")
    except Exception as e:
        logger.error(f"Error al cargar el modelo híbrido: {str(e)}")
        raise

@app.get("/")
async def root():
    """Endpoint raíz que devuelve información sobre la API."""
    return {
        "message": "Sistema de Recomendación de Películas API",
        "version": "1.0.0",
        "endpoints": [
            "/recommend/title/{title}",
            "/recommend/genre/{genre}"
        ]
    }

@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(
    title: str,
    top_n: int = Field(default=10, ge=1, le=50)
):
    """Obtener recomendaciones basadas en el título de una película."""
    try:
        logger.info(f"Buscando recomendaciones para título: {title}")
        recommendations = model.recommend_by_title(title, top_n)
        # Forzar liberación de memoria después de cada recomendación
        gc.collect()
        return RecommendationResponse(
            error=False,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones por título: {str(e)}")
        return RecommendationResponse(
            error=True,
            message=f"Error al procesar la solicitud: {str(e)}"
        )

@app.get("/recommend/genre/{genre}", response_model=RecommendationResponse)
async def recommend_by_genre(
    genre: str,
    top_n: int = Field(default=10, ge=1, le=50)
):
    """Obtener recomendaciones basadas en el género de una película."""
    try:
        logger.info(f"Buscando recomendaciones para género: {genre}")
        recommendations = model.recommend_by_genre(genre, top_n)
        gc.collect()
        return RecommendationResponse(
            error=False,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error al obtener recomendaciones por género: {str(e)}")
        return RecommendationResponse(
            error=True,
            message=f"Error al procesar la solicitud: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
