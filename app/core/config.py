from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Configuraciones de la aplicación."""
    # Configuración de la API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sistema de Recomendación de Películas"
    
    # Configuración de la base de datos
    DATA_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 "proyecto", "data", "movies_filtrado.parquet")
    
    # Configuración del modelo
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Configuración de caché
    CACHE_TTL: int = 3600  # 1 hora en segundos
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Obtener configuración de la aplicación."""
    return Settings() 