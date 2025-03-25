from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Descargar stopwords en español
nltk.download('stopwords')
stopwords_es = stopwords.words('spanish')

# Cargar el DataFrame filtrado
df_filtrado = pd.read_parquet('C:/Users/jugas/OneDrive/Escritorio/Movie recommender/MVP_sistema_recomendacion/proyecto/data/movies_filtrado.parquet')

# Aseguramos que las reseñas no tengan valores nulos
df_filtrado['reseñas'] = df_filtrado['reseñas'].fillna('')

# Vectorizador TF-IDF en español con las stopwords
tfidf = TfidfVectorizer(stop_words=stopwords_es)
tfidf_matrix = tfidf.fit_transform(df_filtrado['reseñas'])

# Calcular la similitud de coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Crear un diccionario de índices de películas
indices = pd.Series(df_filtrado.index, index=df_filtrado['titulo'].str.lower()).drop_duplicates()

# Crear la instancia de FastAPI
app = FastAPI(
    title="Sistema de Recomendación de Películas",
    description="Realiza recomendaciones de películas basadas en reseñas.",
    docs_url="/docs"
)

@app.get('/recomendacion/{titulo}', name="Sistema de recomendación")
async def recomendacion(titulo: str):
    """Recibe un título de película y devuelve las 10 películas más similares basadas en las reseñas."""
    
    # Convertir el título ingresado a minúsculas
    titulo = titulo.strip().lower()

    if titulo not in indices:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Obtener el índice de la película solicitada
    idx = indices[titulo]

    # Calcular las puntuaciones de similitud de coseno
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar las películas por similitud y seleccionar las 10 más similares
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Ignorar la misma película

    # Obtener los índices de las películas recomendadas
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df_filtrado['titulo'].iloc[movie_indices].tolist()

    return {"recomendaciones": recommended_movies}

