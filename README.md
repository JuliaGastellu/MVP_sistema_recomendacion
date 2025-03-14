# 🎬 Movie Recommendation System & API (MVP)

## 📌 Overview  

This project is part of an evaluation task focused on developing a **Minimum Viable Product (MVP)** using **FastAPI**. The system includes endpoints for querying specific data from a movie database and a **personalized recommendation engine** leveraging **machine learning techniques**.  

## 📂 Project Structure  

- **`data/`**: Preprocessed and processed datasets stored in **Parquet** format.  
- **`notebooks/`**: Jupyter Notebooks for different project stages:  
  - `ETL_PI01.ipynb`: Data cleaning, transformation, and loading.  
  - `EDA_PI01.ipynb`: Exploratory Data Analysis to understand patterns and trends.  
  - `ML_modelo.ipynb`: Development and training of the recommendation model.  
- **`main.py`**: FastAPI implementation for data querying and recommendations.  
- **`requirements.txt`**: List of project dependencies.  

## 🔄 Process  

### 🛠️ ETL (Extract, Transform, Load)  

- **Flattened nested columns** and applied transformations according to the requirements.  
- Merged relevant data from the provided CSV files, removing unnecessary columns.  
- Standardized **credit dataset** fields and filtered out irrelevant data.  

### 📊 Exploratory Data Analysis (EDA)  

- Examined dataset attributes (info, data types, null values).  
- **Filtered movies** based on runtime (50-330 mins) to exclude miniseries and anomalies.  
- **Analyzed genres & languages**, keeping only the **top 10 most frequent languages**.  
- **Removed movies released before 1980** to focus on modern audience preferences and optimize resources.  
- **Excluded non-released movies** and movies with **poor ROI calculations**.  
- **Word cloud analysis** on highly rated movies (**score ≥ 7**) to identify common themes.  
- **Finalized with feature selection** and exported the cleaned dataset as a Parquet file.  

### 🎯 Recommendation Model  

The recommendation system is based on **cosine similarity** between categorical attributes like **movie genres, titles, and overview keywords**. After tokenization, the model suggests **5 similar movies** based on content similarity.  

## 🚀 API  

The **FastAPI**-powered **RESTful API** exposes endpoints for querying movie data and generating recommendations.  

🔗 **Live API on Render:** [MVP Recommendation System](https://mvp-sistema-recomendacion.onrender.com/docs)  

### 🔌 API Endpoints  

- **`/cantidad_filmaciones_mes/{mes}`** – Returns the number of movies released in a given month (in Spanish).  
- **`/cantidad_filmaciones_dia/{dia}`** – Returns the number of movies released on a specific weekday (in Spanish).  
- **`/score_titulo/{titulo}`** – Retrieves the movie title, release year, and score.  
- **`/votos_titulo/{titulo}`** – Returns the number of votes and average rating (only if votes > 2000).  
- **`/get_actor/{nombre_actor}`** – Fetches an actor’s total movies, average ROI, and success score.  
- **`/get_director/{nombre_director}`** – Retrieves a director’s movies, release dates, ROI, costs, and revenues.  
- **`/recomendacion/{titulo}`** – Recommends **5 similar movies** based on the input movie title.  

## 👩‍💻 Author  

📧 **Email:** juliacgastellu@gmail.com  
💼 **LinkedIn:** [Julia Gastellu](https://www.linkedin.com/in/julia-gastellu/)  

## 🛠️ Technologies Used  

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="Python" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="40" alt="Pandas" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" height="40" alt="FastAPI" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="40" alt="GitHub" />
</div>
<img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="40" alt="git logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" height="40" alt="vscode logo"  />
</div>

###

<p align="left"></p>

###
