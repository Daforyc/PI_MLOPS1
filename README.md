# PI_MLOPS1
PROYECTO MACHINE LEARNING 1
<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

¡Bienvenidos! 
En esta ocasión, se realizo hacer un trabajo situándome en el rol de un ***MLOps Engineer***.  

<hr>  

## **!HENRY PI01]** !

[Imagen](Henry.jpeg)

## **Descripción del problema (Contexto y rol a desarrollar)**

## Contexto

Tienes tu modelo de recomendación dando unas buenas métricas :smirk:, y ahora, cómo lo llevas al mundo real? :eyes:

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.

En este caso se entrega una base de datos de peliculas para estudiarla limpiarla y finalmente hacer recomendaciones de peliculas segun u criterio de similitud identificado, en este caso el genero de la pelicula

## Rol a desarrollar

Empezaste a trabajar como **`Data Scientist`** en una start-up que provee servicios de agregación de plataformas de streaming. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: un sistema de recomendación que aún no ha sido puesto en marcha! 

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula :sob:): Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevas películas o series, entre otras cosas….  haciendo tu trabajo imposible :weary:. 

Debes empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** (_Minimum Viable Product_) para la próxima semana! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir :exclamation:. Así que te espantas los miedos y te pones manos a la obra :muscle:




# Repositorio MLOPS

Este repositorio contiene un conjunto de funciones y una API desarrollada en Python utilizando el framework FastAPI. Estas funciones permiten realizar consultas relacionadas con películas y obtener información relevante sobre las mismas.

## Configuración del entorno

Asegúrese de tener instaladas las siguientes dependencias:

- fastapi
- pandas
- uvicorn
- numpy
- scikit-learn

Puede instalar estas dependencias ejecutando el siguiente comando:

```
pip install -r requirements.txt
```

## Inicio de la API

Para iniciar la API, ejecute el siguiente comando:

```
python main.py
```

La API estará disponible en `http://localhost:10000`.

## Funciones disponibles

A continuación se describen las funciones disponibles en esta API:

- **Consulta básica**: Puede realizar una consulta de prueba accediendo a la ruta raíz (`/`). Esto retornará un mensaje de bienvenida y proporcionará información adicional sobre las consultas disponibles.

- **Consulta de cantidad de filmaciones por mes**: Puede obtener la cantidad de películas estrenadas en un mes específico históricamente utilizando la ruta `/cantidad_filmaciones_mes/{mes}`. Reemplace `{mes}` con el nombre del mes en español (por ejemplo, `enero`, `febrero`, etc.).

- **Consulta de cantidad de filmaciones por día**: Puede obtener la cantidad de películas estrenadas en un día específico de la semana históricamente utilizando la ruta `/cantidad_filmaciones_dia/{dia}`. Reemplace `{dia}` con el nombre del día en español (por ejemplo, `lunes`, `martes`, etc.).

- **Consulta de puntuación de un título**: Puede obtener la puntuación y la popularidad de una película específica utilizando la ruta `/score_titulo/{titulo}`. Reemplace `{titulo}` con el título de la película que desea consultar.

- **Consulta de votos de un título**: Puede obtener el número total de votos y el promedio de votos de una película específica utilizando la ruta `/votos_titulo/{titulo}`. Reemplace `{titulo}` con el título de la película que desea consultar.

- **Consulta de información de un actor**: Puede obtener la cantidad de filmaciones y el retorno total y promedio de un actor específico utilizando la ruta `/get_actor/{nombre_actor}`. Reemplace `{nombre_actor}` con el nombre del actor que desea consultar.

- **Consulta de información de un director**: Puede obtener el retorno total y la lista de películas de un director específico utilizando la ruta `/get_director/{nombre_director}`. Reemplace `{nombre_director}` con el nombre del director que desea consultar.

- **Recomendación de películas**: Puede obtener una lista de películas recomendadas similares a una película específica utilizando la ruta `/recomendacion/{titulo}`. Reemplace `{titulo}` con el título de la película para la cual desea obtener recomendaciones.

Tenga en cuenta que los nombres de los meses, días de la semana, títulos de películas, actores y directores deben ser ingresados en español.

Para obtener más detalles sobre los parámetros y formatos de respuesta de cada consulta, consulte el código fuente en el archivo `main.py`.

## **`Análisis exploratorio de los datos`**: _(Exploratory Data Analysis-EDA)_

Se realizaron diversas investigaciones y análisis de los datos en el Jupyter Notebook llamado "MLOPS.ipynb". Durante este proceso, se exploraron las relaciones entre las variables del dataset, se identificaron outliers o anomalías y se buscaron patrones interesantes para un análisis posterior.

Para obtener una idea de las palabras más frecuentes en los títulos y utilizarlo en el sistema de recomendación, se emplearon nubes de palabras. Además, se aprovecharon librerías como pandas profiling, missingno, sweetviz y autoviz para obtener conclusiones y visualizaciones útiles.

En el análisis se encontraron 191 películas con calificación perfecta, siendo "Ice Age Columbus: Who were the first Americans?" la primera en la lista. También se generó un histograma para analizar la distribución de las votaciones, revelando que aproximadamente 23,000 películas recibieron una calificación promedio entre 6 y 8 puntos, lo cual representó la mayor acumulación de datos. Cabe destacar que se identificó una gran cantidad de películas con calificación cero.

Además, se visualizaron las películas con mayor y menor revenue, encontrando que "Less Than Zero" es la película con menor revenue, mientras que "Avatar" es la que posee el mayor revenue. Por otro lado, "Revolutionary Girl Utena: The Movie" registró el mayor presupuesto en el dataset.

En cuanto al desempeño a lo largo de los años, se observó que los años con mayor cantidad de contenidos publicados fueron 2014 con 1,974 películas y 2015 con 1,905 películas. Por otro lado, se identificaron varios años con solo una película publicada.

Los promedios obtenidos entre las tablas fueron los siguientes:

- Revenue: 11,207,870
- Runtime: 94.13
- Vote Average: 5.62
- Release Year: 1991.88
- Return: 658.74

Asimismo, se utilizó el profiling para identificar otros aspectos e interacciones en el dataset mediante df.profile_report().

Entre los hallazgos, se destacó que los géneros más repetidos fueron el drama y la comedia. En cuanto a los idiomas, se encontraron 2,438 películas en inglés y 32,269 en francés. La compañía más frecuente fue Metro-Golding Mayer con 742 apariciones, seguida de Paramount. Como era de esperar, Estados Unidos fue el país de origen con 17,851 películas.

En relación a las correlaciones, se encontraron patrones marcados entre el retorno, el revenue y la popularidad, así como entre el presupuesto y la calificación promedio. Esto se evidenció a través de heatmaps construidos con la ayuda de Seaborn.

## **`Sistema de recomendacion (MLOPS)`**:

![SR](m.gif)

El sistema de recomendación implementado tiene como objetivo sugerir películas similares en función de un título ingresado. Para lograr esto, se sigue un proceso técnico que involucra la organización de los datos y la generación de una matriz de similitud utilizando los géneros y los títulos de las películas.

En primer lugar, se restringe el dataset a las columnas relevantes para el cálculo de similitud, que son 'genres', 'title' e 'id'. A continuación, se realiza un preprocesamiento en la columna 'genres' para convertir los datos en dummies correspondientes a cada género presente en una película.

Una vez obtenidos los dummies de los géneros, se identifica el título seleccionado y se busca su género correspondiente en el conjunto de datos. Luego, se concatena esta información con la matriz de similitud previamente generada, la cual se obtiene a partir de los dummies de géneros.

A continuación, se aplica el algoritmo Kvecinos (K-Nearest Neighbors) utilizando el vecino más cercano. Este algoritmo se utiliza para encontrar las películas más similares en función de la matriz de similitud generada. El valor de k se establece en 6 para obtener un conjunto de recomendaciones.

Después de ajustar el modelo Kvecinos al conjunto de datos, se ordenan las películas similares según su calificación promedio. Esto permite presentar las películas más relevantes y mejor valoradas como sugerencias.

Finalmente, se muestra una lista de recomendaciones limitada a un máximo de 5 películas. Estas recomendaciones se generan utilizando la función `enumerate` en un bucle `for`, descartando la película con el mismo título ingresado.

Esta función de recomendación permite a los usuarios descubrir películas similares en base a sus títulos, lo cual puede ser útil para explorar nuevas opciones de entretenimiento.

## Contacto
![Escribir](Edit.gif)


Para cualquier pregunta o consulta adicional, no dude en ponerse en contacto con el autor del repositorio:

![Crecer](9TLY.gif)


- Email: david.foca@hotmail.com
- Github: Daforyc - [https://github.com/Daforyc](https://github.com/Dafory
