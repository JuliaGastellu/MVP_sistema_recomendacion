from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Descargar stopwords en español
nltk.download('stopwords')
stopwords_es = stopwords.words('spanish')

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'proyecto', 'data', 'movies_filtrado.parquet')

# Modelos Pydantic para validación
class MovieRecommendation(BaseModel):
    titulo: str
    sinopsis: str
    puntuacion: float

class GenreRecommendation(MovieRecommendation):
    generos: str

class ErrorResponse(BaseModel):
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Caché para recomendaciones
recommendation_cache = {}
CACHE_TTL = 3600  # 1 hora en segundos

def clean_title(title: str) -> str:
    """Limpia el título para búsqueda."""
    return re.sub(r'[^\w\s]', '', title.lower().strip())

def find_similar_titles(title: str, df: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    """Encuentra títulos similares usando similitud de texto."""
    clean_query = clean_title(title)
    titles = df['titulo'].apply(clean_title).tolist()
    
    # Crear vectorizador para títulos
    title_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
    title_matrix = title_vectorizer.fit_transform(titles + [clean_query])
    
    # Calcular similitud
    similarities = cosine_similarity(title_matrix[-1:], title_matrix[:-1])[0]
    
    # Encontrar títulos similares
    similar_indices = np.where(similarities >= threshold)[0]
    return [df.iloc[i]['titulo'] for i in similar_indices]

@lru_cache(maxsize=100)
def get_cached_recommendations(cache_key: str, recommendation_type: str):
    """Obtiene recomendaciones del caché si están disponibles y no han expirado."""
    if cache_key in recommendation_cache:
        data, timestamp = recommendation_cache[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_TTL):
            logger.info(f"Recomendación encontrada en caché: {cache_key}")
            return data
    return None

def cache_recommendation(cache_key: str, data: List[Dict[str, Any]]):
    """Almacena recomendaciones en el caché."""
    recommendation_cache[cache_key] = (data, datetime.now())
    logger.info(f"Recomendación almacenada en caché: {cache_key}")

try:
    # Cargar el DataFrame filtrado
    df_filtrado = pd.read_parquet(data_path)
    df_filtrado['reseñas'] = df_filtrado['reseñas'].fillna('')
    
    # Vectorizador TF-IDF en español con stopwords
    tfidf = TfidfVectorizer(stop_words=stopwords_es)
    tfidf_matrix = tfidf.fit_transform(df_filtrado['reseñas'])
    
    # Calcular similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Crear un diccionario de índices de películas
    indices = pd.Series(df_filtrado.index, index=df_filtrado['titulo'].str.lower()).drop_duplicates()
    
    # Cargar el modelo SentenceTransformer para embeddings de géneros
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_movie_vector(genres, model):
        genres_text = " ".join(genres)
        return model.encode([genres_text])[0]
    
    # Preprocesar los géneros y calcular los vectores de género
    df_filtrado["generos"] = df_filtrado["generos"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
    df_filtrado["vector"] = df_filtrado["generos"].apply(lambda x: get_movie_vector(x, model))
    
    logger.info("Datos y modelos cargados correctamente")
except Exception as e:
    logger.error(f"Error al cargar datos o modelos: {str(e)}")
    raise

# Crear la instancia de FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="Recomendaciones basadas en reseñas y géneros.",
    docs_url="/docs"
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
            detail="Error interno del servidor. Por favor, inténtelo de nuevo más tarde."
        ).dict()
    )

@app.get('/recomendacion/{titulo}', response_model=List[MovieRecommendation])
async def recomendacion(
    titulo: str,
    top_n: int = Query(5, ge=1, le=20, description="Número de recomendaciones a devolver")
):
    """Devuelve las películas más similares basadas en reseñas."""
    try:
        # Verificar caché
        cache_key = f"review_{titulo}_{top_n}"
        cached_result = get_cached_recommendations(cache_key, "review")
        if cached_result:
            return cached_result
        
        titulo = titulo.strip().lower()
        
        # Buscar títulos similares si no se encuentra exactamente
        if titulo not in indices:
            similar_titles = find_similar_titles(titulo, df_filtrado)
            if similar_titles:
                raise HTTPException(
                    status_code=404,
                    detail=f"Película no encontrada. ¿Quizás quisiste decir: {', '.join(similar_titles[:3])}?"
                )
            else:
                raise HTTPException(status_code=404, detail="Película no encontrada")
        
        idx = indices[titulo]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recomendaciones = df_filtrado.iloc[movie_indices][["titulo", "sinopsis", "puntuacion"]]
        result = recomendaciones.to_dict(orient='records')
        
        # Almacenar en caché
        cache_recommendation(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en recomendacion: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get('/recomendacion_genero/{titulo}', response_model=List[GenreRecommendation])
async def recomendacion_genero(
    titulo: str,
    top_n: int = Query(5, ge=1, le=20, description="Número de recomendaciones a devolver")
):
    """Devuelve películas similares basadas en los géneros."""
    try:
        # Verificar caché
        cache_key = f"genre_{titulo}_{top_n}"
        cached_result = get_cached_recommendations(cache_key, "genre")
        if cached_result:
            return cached_result
        
        # Buscar película
        movie_row = df_filtrado[df_filtrado['titulo'].str.contains(titulo, case=False, na=False)]
        if movie_row.empty:
            similar_titles = find_similar_titles(titulo, df_filtrado)
            if similar_titles:
                raise HTTPException(
                    status_code=404,
                    detail=f"Película no encontrada. ¿Quizás quisiste decir: {', '.join(similar_titles[:3])}?"
                )
            else:
                raise HTTPException(status_code=404, detail="Película no encontrada")
        
        movie_vector = movie_row.iloc[0]["vector"].reshape(1, -1)
        similarities = cosine_similarity(movie_vector, np.stack(df_filtrado["vector"].values))
        df_filtrado["similarity"] = similarities[0]
        
        recomendaciones = df_filtrado.sort_values(by="similarity", ascending=False).head(top_n)[["titulo", "generos", "sinopsis", "puntuacion"]]
        recomendaciones["generos"] = recomendaciones["generos"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
        
        result = recomendaciones.to_dict(orient='records')
        
        # Almacenar en caché
        cache_recommendation(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en recomendacion_genero: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get('/buscar/{query}')
async def buscar_peliculas(
    query: str,
    limit: int = Query(5, ge=1, le=20, description="Número máximo de resultados")
):
    """
    Busca películas por título o descripción.
    
    Este endpoint permite buscar películas en la base de datos utilizando palabras clave.
    La búsqueda se realiza tanto en los títulos como en las sinopsis de las películas.
    
    Parámetros:
    - query: Palabra o frase para buscar
    - limit: Número máximo de resultados a devolver (entre 1 y 20)
    
    Retorna:
    - Un objeto JSON con los resultados de la búsqueda
    """
    try:
        # Validar que la consulta no esté vacía
        if not query or query.strip() == "":
            return {
                "error": False,
                "mensaje": "La consulta de búsqueda no puede estar vacía",
                "resultados": []
            }
            
        query = query.strip().lower()
        
        # Asegurarse de que las columnas existan y sean del tipo correcto
        if 'titulo' not in df_filtrado.columns or 'sinopsis' not in df_filtrado.columns:
            logger.error("Columnas 'titulo' o 'sinopsis' no encontradas en el DataFrame")
            return {
                "error": True,
                "mensaje": "Error en la estructura de datos",
                "detalle": "Columnas necesarias no encontradas en el dataset"
            }
        
        # Convertir columnas a string si no lo son ya
        df_filtrado['titulo'] = df_filtrado['titulo'].astype(str)
        df_filtrado['sinopsis'] = df_filtrado['sinopsis'].astype(str)
        
        # Buscar por título (prioridad alta)
        title_matches = df_filtrado[df_filtrado['titulo'].str.lower().str.contains(query, na=False)]
        
        # Buscar por sinopsis (prioridad baja)
        synopsis_matches = df_filtrado[df_filtrado['sinopsis'].str.lower().str.contains(query, na=False)]
        
        # Combinar resultados, dando prioridad a los títulos
        results = pd.concat([title_matches, synopsis_matches]).drop_duplicates().head(limit)
        
        # Imprimir información de depuración
        logger.info(f"Búsqueda para: '{query}'")
        logger.info(f"Resultados encontrados: {len(results)}")
        logger.info(f"Títulos encontrados: {len(title_matches)}")
        logger.info(f"Sinopsis encontradas: {len(synopsis_matches)}")
        
        if results.empty:
            # Buscar títulos similares para sugerencias
            similar_titles = find_similar_titles(query, df_filtrado)
            if similar_titles:
                return {
                    "error": False,
                    "mensaje": "No se encontraron películas exactas",
                    "sugerencias": similar_titles[:3],
                    "resultados": []
                }
            else:
                return {
                    "error": False,
                    "mensaje": "No se encontraron películas",
                    "resultados": []
                }
        
        # Preparar resultados
        formatted_results = []
        for _, row in results.iterrows():
            # Asegurarse de que los campos existan y tengan valores por defecto
            titulo = row.get("titulo", "Sin título")
            sinopsis = row.get("sinopsis", "Sin sinopsis")
            puntuacion = float(row.get("puntuacion", 0.0))
            
            # Manejar géneros de manera segura
            generos = row.get("generos", [])
            if isinstance(generos, list):
                generos_str = ", ".join(generos)
            else:
                generos_str = str(generos)
            
            formatted_results.append({
                "titulo": titulo,
                "sinopsis": sinopsis,
                "puntuacion": puntuacion,
                "generos": generos_str
            })
        
        return {
            "error": False,
            "mensaje": f"Se encontraron {len(results)} películas",
            "resultados": formatted_results
        }
    except Exception as e:
        logger.error(f"Error en búsqueda: {str(e)}")
        # Devolver una respuesta JSON con información sobre el error
        return {
            "error": True,
            "mensaje": "Error al procesar la búsqueda",
            "detalle": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 