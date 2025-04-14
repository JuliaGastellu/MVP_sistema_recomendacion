from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import logging
import sys
import os
import nltk
import gc
import tempfile

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
class MovieRecommendation(BaseModel):
    """Modelo para una película recomendada."""
    titulo: str
    sinopsis: str
    puntuacion: float
    generos: List[str]
    anio_estreno: int

    model_config = {
        "json_schema_extra": {  # Updated from schema_extra
            "example": {
                "titulo": "Ejemplo de película",
                "sinopsis": "Esta es una sinopsis de ejemplo",
                "puntuacion": 8.5,
                "generos": ["Acción", "Aventura"],
                "anio_estreno": 2023
            }
        }
    }

class RecommendationResponse(BaseModel):
    """Modelo para la respuesta de recomendaciones."""
    error: bool
    message: Optional[str] = None
    recommendations: Optional[List[MovieRecommendation]] = None
    suggestions: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {  # Updated from schema_extra
            "example": {
                "error": False,
                "message": None,
                "recommendations": [
                    {
                        "titulo": "Ejemplo de película",
                        "sinopsis": "Esta es una sinopsis de ejemplo",
                        "puntuacion": 8.5,
                        "generos": ["Acción", "Aventura"],
                        "anio_estreno": 2023
                    }
                ],
                "suggestions": ["Película 1", "Película 2"]
            }
        }
    }

# Variable global para el modelo
model = None

# At the top with other imports
import warnings
warnings.filterwarnings('ignore')

# At the top of the file, after other imports
import tempfile

# Set up environment variables for model caching
cache_dir = tempfile.mkdtemp()
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    global model
    try:
        logger.info("Iniciando carga del modelo híbrido...")
        gc.collect()
        
        try:
            # Try multiple possible data paths for different environments
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'proyecto', 'data', 'movies_filtrado.parquet'),
                os.path.join(os.getcwd(), 'proyecto', 'data', 'movies_filtrado.parquet'),
                os.path.join('/opt/render/project/src', 'proyecto', 'data', 'movies_filtrado.parquet')
            ]
            
            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
                    
            if not data_path:
                raise FileNotFoundError(f"No se encuentra el archivo de datos en ninguna ubicación conocida")
                
            logger.info(f"Usando archivo de datos en: {data_path}")
            model = load_hybrid_model()
            
        except ImportError as e:
            logger.error(f"Error de importación al cargar el modelo: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error general al cargar el modelo: {str(e)}")
            raise
            
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
    top_n: int = Query(default=10, ge=1, le=50)  # Changed from Field to Query
):
    """Obtener recomendaciones basadas en el título de una película."""
    try:
        logger.info(f"Buscando recomendaciones para título: {title}")
        recommendations = model.recommend_by_title(title, top_n)
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
    top_n: int = Query(default=10, ge=1, le=50)  # Changed from Field to Query
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

# Update the main block to handle port from environment
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)  # Changed reload to False for production
