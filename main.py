from fastapi import FastAPI
#from fastapi.responses import JSONResponse
#from fastapi.encoders import jsonable_encoder
import pandas as pd
import uvicorn 
import numpy as np
from typing import Optional
import ast
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer


app = FastAPI(title = 'MLOPS')
df=pd.read_csv("moviestrasnf2.csv")

# introduccion
@app.get("/")
async def index():
    return "Hola! aqui puedes realizar consultas para peliculas, para mas informacion ir a /docs"

@app.get("/")
def presentacion():
    return {"PI 01 - David Arturo Fory Castrillon"}

@app.get("/contacto")
def contacto():
    return "Email: david.foca@hotmail.com / Github: Daforyc - https://github.com/Daforyc -"

@app.get("/menu")
def menu():
    return "Las funciones utilizadas: peliculas_mes, peliculas_dis, franquicia, peliculas_pais, productoras, retorno, recomendacion"

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    df = pd.read_csv("moviestrasnf2.csv")
    meses = {"enero": 1 ,"febrero": 2 ,"marzo": 3 ,"abril": 4 ,"mayo": 5 ,"junio": 6 ,"julio": 7 ,"agosto": 8 ,"septiembre": 9 ,"octubre": 10 ,"noviembre": 11 ,"diciembre": 12 }
    if mes not in meses:
        return "Invalid month"
    lect = meses[mes]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    cuenta =(df['release_date'].dt.month == lect).value_counts()[True]
    return {'mes': str(mes), 'cantidad': int(cuenta)}  

@app.get('/peliculas_dis/{dis}')
def peliculas_dis(weekday: str):
    '''Se ingresa el día y la función retorna la cantidad de películas que se estrenaron ese día históricamente'''
    # Diccionario de equivalencias de días de la semana en inglés y español
    ndia=weekday
    dias_semana = {
        'lunes': 'Monday',
        'martes': 'Tuesday',
        'miércoles': 'Wednesday',
        'jueves': 'Thursday',
        'viernes': 'Friday',
        'sábado': 'Saturday',
        'domingo': 'Sunday'
    }
    # Verificar si se ingresó un día en español y obtener su equivalente en inglés
    if weekday.lower() in dias_semana:
        weekday = dias_semana[weekday.lower()]
    
    # Cargar el dataset
    df = pd.read_csv("moviestrasnf2.csv")
    # Convertir la columna release_date a un objeto datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # Filtrar las filas que no tienen un valor nulo en la columna release_date
    df = df[df['release_date'].notnull()]
    # Obtener el nombre del día de la semana para cada fecha en la columna release_date
    df['weekday'] = df['release_date'].dt.day_name()
    # Filtrar por el nombre del día de la semana buscado y contar la cantidad de películas
    count = len(df[df['weekday'].str.lower() == weekday.lower()])
    # Retornar el resultado en un diccionario con el formato {'dia': weekday, 'cantidad': count}
    return {'dia': str(ndia), 'cantidad': int(count)}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo:str):
    df = pd.read_csv("movietrasnf2.csv", na_values=["", "NaN", "NA"], dtype={"title": str})
    df["title"] = df["title"].fillna("") 
    df = df[df["title"].str.contains(titulo, case=False)] 
    if df.empty: 
        return None
    else:
        pelicula = df.iloc[0]["title"]
        anio = df.iloc[0]["release_year"]
        vote_average = df['vote_average'].values[0]
        popularity = df['popularity'].values[0]
        score_pop = vote_average / popularity
    return {'titulo':str(titulo), 'anio':int(anio), 'popularidad':float(score_pop)}

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo:str):
    df = pd.read_csv("datasets/movietrasnf2.csv", na_values=["", "NaN", "NA"], dtype={"title": str})
    df["title"] = df["title"].fillna("") 
    df = df[df["title"].str.contains(titulo, case=False)] 
    if df.empty: 
        return None
    else:
        pelicula = df.iloc[0]["title"]
        votos = df.iloc[0]["vote_count"]
        vote_average = df.iloc[0]["vote_average"]
        anio = df.iloc[0]["release_year"]
        if votos < 2000:
            return {'la pelicula':pelicula, 'no posee más de 2000 votaciones solo tiene':votos}
        return {'titulo':str(pelicula), 'anio':int(anio),'voto_total':int(votos), 'voto_promedio':float(vote_average)}
    
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor:str):
    # filtra las filas que corresponden a la franquicia de entrada y no son NaN
    df = pd.read_csv("movietrasnf2.csv")
    actor_df = df[(df['cast'].notna()) & (df['cast'].str.contains(nombre_actor))]
    
    # si no hay filas correspondientes a la franquicia, retorna un mensaje
    if actor_df.empty:
        return {'No se encontró la actor con el nombre':str(nombre_actor)}
    
    # cuenta la cantidad de filas y suma los ingresos de la franquicia
    cantidad = actor_df.shape[0]
    ganancia_total = actor_df['revenue'].sum()
    gpromedio= ganancia_total/cantidad
    return {'actor':str(nombre_actor), 'cantidad_filmaciones':int(cantidad),'retorno_total':int(ganancia_total), 'retorno_promedio':float(gpromedio)}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    df = pd.read_csv("movietrasnf2.csv")
    director_movies = df[df['director_name'] == nombre_director]
    total_return = director_movies['return'].sum()
    movie_info = director_movies[['title', 'release_date', 'return', 'budget', 'revenue']]
    movies = []
    for _, row in movie_info.iterrows():
        movie = {
            'pelicula': row['title'],
            'anio': ['release_date'],
            'retorno_pelicula': row['return'],
            'budget_pelicula': row['budget'],
            'revenue_pelicula': row['revenue']
            }
        movies.append(movie)
    return {
        'director': nombre_director,
        'retorno_total_director': int(total_return),
        'peliculas': list(movies)
        }

# ML
df2 = pd.read_csv("moviestrasnf2.csv")
df2['genres'] = df2['genres'].apply(ast.literal_eval)
generos_df = df2['genres'].str.get_dummies('|')
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
#    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    if titulo not in df2['title'].values:
        return {"Título inválido"}
    selected_title = titulo
    selected_genres = df2.loc[df2['title'] == selected_title]['genres'].values[0]
    df2['genre_similarity'] = df2['genres'].apply(lambda x: len(set(selected_genres) & set(x)) / len(set(selected_genres) | set(x)))
    df2['same_series'] = df2['title'].apply(lambda x: 1 if pd.notnull(x) and titulo in x else 0)
    features_df = pd.concat([generos_df, df2['vote_average'], df2['genre_similarity'], df2['same_series']], axis=1)
    
    # Preprocesamiento para manejar valores NaN en features_df
    imputer = SimpleImputer(strategy='mean')
    features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)
    
    k = 6
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
    knn.fit(features_df)
    indices = knn.kneighbors(features_df.loc[df2['title'] == selected_title])[1].flatten()
    recommended_movies = list(df2.iloc[indices]['title'])
    
    selected_score = df2.loc[df2['title'] == selected_title]['vote_average'].values[0]
    recommended_movies = sorted(recommended_movies, key=lambda x: (df2.loc[df2['title'] == x]['same_series'].values[0], df2.loc[df2['title'] == x]['vote_average'].values[0], df2.loc[df2['title'] == x]['genre_similarity'].values[0]), reverse=True)
    recommended_movies = [movie for movie in recommended_movies if movie != selected_title]
    
    recommended_movies_info = []
    for i, pelicula in enumerate(recommended_movies[:5]):
        score = df2.loc[df2['title'] == pelicula]['vote_average'].values[0]
        genres = df2.loc[df2['title'] == pelicula]['genres'].values[0]
        gen_str = ', '.join(genres)
        recommended_movies_info.append({
            'title': pelicula,
            'genres': gen_str,
            'score': score
        })
        if i == 4:
            break
    return {'lista_recomendada': recommended_movies_info}

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")
#uvicorn.run(app, host="0.0.0.0", port=10000) 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)