from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np

# Descargar stopwords en español
nltk.download('stopwords')
stopwords_es = stopwords.words('spanish')

# Cargar el DataFrame filtrado
df_filtrado = pd.read_parquet('C:/Users/jugas/OneDrive/Escritorio/Movie recommender/MVP_sistema_recomendacion/proyecto/data/movies_filtrado.parquet')

# Asegurar que las reseñas no tengan valores nulos
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

# Crear la instancia de FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="Recomendaciones basadas en reseñas y géneros.",
    docs_url="/docs"
)

@app.get('/recomendacion/{titulo}', name="Sistema de recomendación por reseñas")
async def recomendacion(titulo: str, top_n: int = 5):
    """Devuelve las 10 películas más similares basadas en reseñas."""
    titulo = titulo.strip().lower()
    if titulo not in indices:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    idx = indices[titulo]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    # Devuelve las recomendaciones con título, sinopsis y puntuación
    recomendaciones = df_filtrado.iloc[movie_indices][["titulo", "sinopsis", "puntuacion"]]
    return recomendaciones.to_dict(orient='records')

@app.get('/recomendacion_genero/{titulo}', name="Sistema de recomendación por géneros")
async def recomendacion_genero(titulo: str, top_n: int = 5):
    """Devuelve películas similares basadas en los géneros."""
    movie_row = df_filtrado[df_filtrado['titulo'].str.contains(titulo, case=False, na=False)]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    movie_vector = movie_row.iloc[0]["vector"].reshape(1, -1)
    similarities = cosine_similarity(movie_vector, np.stack(df_filtrado["vector"].values))
    df_filtrado["similarity"] = similarities[0]
    
    # Devuelve las recomendaciones con título, géneros, sinopsis y puntuación
    recomendaciones = df_filtrado.sort_values(by="similarity", ascending=False).head(top_n)[["titulo", "generos", "sinopsis", "puntuacion"]]
    
    # Asegurarse de que los géneros sean cadenas de texto antes de devolverlas
    recomendaciones["generos"] = recomendaciones["generos"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    
    return recomendaciones.to_dict(orient='records')

# Ejecutar la API
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
