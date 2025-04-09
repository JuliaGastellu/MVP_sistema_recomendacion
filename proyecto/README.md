# 🎬 Sistema de Recomendación de Películas

Este proyecto implementa un sistema de recomendación de películas utilizando un modelo híbrido que combina TF-IDF y Sentence Transformers. El sistema proporciona una API REST para obtener recomendaciones basadas en diferentes criterios.

## ✨ Características

- 🎯 Recomendaciones basadas en título de película
- 🎭 Recomendaciones por género
- 📅 Recomendaciones por año de lanzamiento
- 🔍 Búsqueda de películas por título o descripción
- 🤖 Modelo híbrido que combina TF-IDF y Sentence Transformers
- 🚀 API REST con FastAPI
- 📚 Documentación automática de la API
- 🛡️ Manejo de errores robusto
- 📝 Logging completo

## 📋 Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crear y activar un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Descargar recursos de NLTK:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 📁 Estructura del Proyecto

```
├── proyecto/
│   ├── data/
│   │   ├── movies.parquet
│   │   └── movies_filtrado.parquet
│   ├── images/
│   │   ├── endpoint1.png
│   │   └── endpoint2.png
│   ├── models/
│   │   ├── hybrid_model.py
│   │   ├── sentence_transformer_model.py
│   │   └── tfidf_model.py
│   └── notebooks/
│       ├── EDA.ipynb
│       ├── ETL.ipynb
│       ├── TF-IDF-Cosine-Sim_model.ipynb
│       └── all-MiniLM-L6-v2_model.ipynb
├── README.md
├── main.py
├── requirements.txt
└── .gitignore
```

## 💻 Uso

1. Iniciar el servidor:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. Acceder a la documentación de la API:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔌 Endpoints de la API

### GET /
- Descripción: Información básica sobre la API
- Respuesta: Lista de endpoints disponibles

### GET /recommend/title/{title}
- Descripción: Obtiene recomendaciones basadas en el título de una película
- Parámetros:
  - title: Título de la película
  - top_n: Número de recomendaciones (default: 10)

### GET /recommend/genre/{genre}
- Descripción: Obtiene recomendaciones basadas en el género
- Parámetros:
  - genre: Género de la película
  - top_n: Número de recomendaciones (default: 10)

### GET /recommend/year/{year}
- Descripción: Obtiene recomendaciones basadas en el año
- Parámetros:
  - year: Año de lanzamiento
  - top_n: Número de recomendaciones (default: 10)

### GET /search/{query}
- Descripción: Busca películas por título o descripción
- Parámetros:
  - query: Término de búsqueda
  - limit: Número máximo de resultados (default: 10)

## 📊 Ejemplo de Respuesta

```json
{
  "error": false,
  "recommendations": [
    {
      "titulo": "Matrix",
      "sinopsis": "Un programador descubre...",
      "puntuacion": 8.7,
      "generos": ["Ciencia Ficción", "Acción"],
      "similitud": 0.95
    }
  ]
}
```

## 🤖 Modelo Híbrido

El sistema utiliza un modelo híbrido que combina:

1. **TF-IDF**:
   - Vectorización de texto
   - Ponderación de términos
   - Stemming para mejor coincidencia

2. **Sentence Transformers**:
   - Modelo preentrenado 'all-MiniLM-L6-v2'
   - Captura de significado semántico
   - Vectores de alta dimensionalidad

## 🤝 Contribuir

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles. 