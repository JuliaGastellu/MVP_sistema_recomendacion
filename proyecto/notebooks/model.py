from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Descargar las stopwords en español
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Ruta del archivo Parquet
file_path = 'C:/Users/jugas/OneDrive/Escritorio/Movie recommender/MVP_sistema_recomendacion/proyecto/data/movies_filtrado.parquet'

# Leer el archivo Parquet
df_filtrado = pd.read_parquet(file_path)

# Crear una columna 'caracteristicas' combinando géneros, reseñas y sinopsis
df_filtrado['caracteristicas'] = df_filtrado['generos'].astype(str) + ' ' + \
                                  df_filtrado['reseñas'].fillna('') + ' ' + \
                                  df_filtrado['sinopsis'].fillna('')

# Convertir la columna 'caracteristicas' a tipo texto
df_filtrado['caracteristicas'] = df_filtrado['caracteristicas'].astype(str)

# Eliminar filas con valores faltantes en 'caracteristicas'
df_filtrado = df_filtrado.dropna(subset=['caracteristicas'])

# Asegurarse de que todas las entradas en la columna 'reseñas' sean cadenas de texto
df_filtrado['reseñas'] = df_filtrado['reseñas'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Usamos TfidfVectorizer con las stopwords en español para procesar las reseñas
tfidf = TfidfVectorizer(stop_words=spanish_stopwords)
tfidf_matrix = tfidf.fit_transform(df_filtrado['reseñas'])

# Calcular la similitud del coseno entre las películas basándonos en las características
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Función para obtener las películas más similares
def obtener_recomendaciones(titulo_pelicula, cosine_sim=cosine_sim):
    # Verificar si el título de la película existe en el DataFrame
    if titulo_pelicula not in df_filtrado['titulo'].values:
        return f'La película "{titulo_pelicula}" no se encuentra en el DataFrame.'
    
    # Obtengo el índice de la película
    idx = df_filtrado.index[df_filtrado['titulo'] == titulo_pelicula].tolist()[0]
    
    # Obtengo las puntuaciones de similitud con todas las películas
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordeno las películas basándome en las similitudes, y obtengo los 10 más similares
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    # Obtengo los índices de las películas más similares
    movie_indices = [i[0] for i in sim_scores]
    
    # Devuelvo las películas más similares
    return df_filtrado['titulo'].iloc[movie_indices]

# Ejemplo de uso
titulo_ejemplo = 'Origen'  # Título de la película de ejemplo
recomendaciones = obtener_recomendaciones(titulo_ejemplo)
print(f'Recomendaciones para "{titulo_ejemplo}":\n', recomendaciones)
