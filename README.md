<div id="header" align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXVoM3BiOTBmd3d1YWcyMWg4dGMzZXU1Y2IwdXRnYmMxN2g1MmZoZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/EZr27ZbJwmjE9PGyLN/giphy.gif" width="200"/>
  </div>

<br>

<h1 align="center">
  <b>Hi, I'm</b>
  <b style="color: #22b05d;">Jair Trujillo Rincon</b>
  <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="35">
</h1>

<p align="center">
  <a href="https://github.com/DenverCoder1/readme-typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Roboto&color=32CD32&size=25&center=true&vCenter=true&width=900&height=100&lines=I'm+a+Financial+Administrator+by+profession+💼;I've+focused+my+career+on+Data+Engineering+and+Data+Science+🛠️📊;Passionate+about+Designing+and+Optimizing+Scalable+Data+Pipelines+🔄📈;Machine+Learning+and+Deep+Learning+Models+🤖🧠;Advanced+Analytics+Solutions+📊✨;Self-taught+Data+Learner+with+a+Growth+Mindset+📚🚀">
  </a>
</p>

## 🙋 About Me

I have been learning about data analysis for over three years. I began with data analysis and later expanded into data science, machine learning, and deep learning. As my models grew more sophisticated, I realized the importance of efficiently processing large volumes of data, which led me to study data engineering. Today, I focus on building robust, end-to-end analytical solutions.

<!-- * 🖥️  See my portfolio at [miportafolio](http://app) Aqui puedo agregar los dashboards de power bi -->
* 🌍  I'm based in Medellin, Colombia
* ✉️  You can contact me at [**coremprejair@gmail.com**](mailto:coremprejair@gmail.com)
* 🚀  I'm currently working on **Data Science**
* 🧠  I'm currently enhancing my skills in **Artificial Intelligence with Python**
* 👥  I'm looking to collaborate on **Analytics** and **Data Science** projects
* 💬  Ask me about free courses and publicly available data sources

## 🎰 Proyecto: Predicción de juegos de azar (Colombia)

Este repositorio ahora incluye un pipeline base en Python para:

1. **Extraer datos** desde una fuente oficial de datos abiertos en Colombia.
2. **Normalizar resultados** en un esquema común (`draw_date`, `strategy`, `game`, `number`, `prize_type`).
3. **Entrenar modelos por estrategia/juego**.
4. **Generar predicción diaria** por cada estrategia disponible.

### Estructura agregada

- `/src/lottery_pipeline/config.py`: configuración de fuentes oficiales.
- `/src/lottery_pipeline/collector.py`: ingesta y normalización.
- `/src/lottery_pipeline/modeling.py`: entrenamiento y predicción.
- `/src/lottery_pipeline/cli.py`: interfaz de línea de comandos.

La ingesta usa la API Socrata de Datos Abiertos (`/resource/{dataset}.json`).
La fuente configurada es el dataset oficial **Resultados de los sorteos
loterías** (`i3kx-3zps`). Sus campos se detectan de forma flexible y se
normalizan al esquema interno. Si Datos Abiertos cambia el identificador, se
puede reemplazar la fuente mediante `LOTTERY_SOURCES_JSON` sin modificar el
código.

### Uso rápido

```bash
uv sync --group dev

# 1) Extraer datos
PYTHONPATH=src uv run python -m lottery_pipeline collect --limit 50000

# 2) Entrenar y predecir
PYTHONPATH=src uv run python -m lottery_pipeline train --min-rows 30

# 3) Validar como futuro sobre histórico (walk-forward)
PYTHONPATH=src uv run python -m lottery_pipeline backtest --min-rows 30
```

### Archivos de salida

- `data/raw/lottery_results.csv`: resultados normalizados.
- `data/raw/metadata.json`: metadatos de extracción.
- `models/lottery_models.pkl`: modelos entrenados por estrategia/juego.
- `predictions/daily_predictions.csv`: predicción para la fecha objetivo.

El repositorio incluye `data/lottery.db` como semilla histórica. La copia
actual contiene 1.024 resultados de Lotería de Medellín entre 2007-01-05 y
2026-10-09. El workflow diario restaura y actualiza esa base mediante cache;
los nuevos resultados se consolidan con `ON CONFLICT`, sin borrar el histórico.

### Datos, frecuencia y aprendizaje

Cada modelo se entrena por separado para cada combinación `strategy/game`, usando
el histórico del número principal de cada sorteo: fecha, número ganador,
rezagos (`lag1`, `lag7`), media móvil de siete sorteos y variables de calendario.
Los premios secundarios se conservan en SQLite y no se duplican como observaciones
de entrenamiento; se usan para evaluar si el número predicho coincide con cualquier
premio publicado.
Los juegos diarios, semanales y mensuales no se mezclan. La predicción solo se
genera cuando la fecha objetivo coincide con el patrón de sorteos observado.

El modelo actual es supervisado (`RandomForestRegressor` con suavizado
histórico), no aprendizaje reforzado. Cada predicción conserva una probabilidad
histórica para su combinación de estrategia y juego. Las evaluaciones calculan
probabilidad observada de acertar cualquier premio, el premio mayor, premios
secundarios, `hit@3`, `hit@5` y Brier por estrategia/juego; sirven para monitorear
calibración y rendimiento, pero no se usan como una falsa señal de recompensa.
En sorteos aleatorios no existe una acción controlable ni una recompensa estable
que justifique afirmar que un agente puede aprender una política ganadora.

La fuente oficial identifica categorías como `Mayor` y `Seco`. El pipeline compara
el número predicho con el número publicado en cada categoría y registra los tipos
coincidentes. Esto evalúa aciertos exactos; reglas adicionales como aproximaciones,
series o terminales requieren que la fuente publique esos campos y una definición
oficial de la regla antes de calcularlas.

### Comunicación y aprendizaje

Las alertas se envían en HTML y texto plano. Un correo agrupa todos los aciertos
pendientes de la ejecución, muestra una tabla con fecha, estrategia, juego, número
y categoría del premio, y separa el resumen entre premios mayores y secundarios.
Esto evita enviar un correo por cada coincidencia y conserva compatibilidad con
clientes que no renderizan HTML.

El sistema actual no es aprendizaje reforzado: no existe una acción sobre el
sorteo ni una recompensa que pueda modificar sus probabilidades reales. Los
“pesos” del modelo son: la frecuencia histórica suavizada de cada número y un
factor exponencial según su distancia a la predicción del Random Forest. El
Random Forest sí aprende pesos internos de sus árboles, pero no se actualizan con
los aciertos del correo. La mejora correcta es recalibrar y comparar estrategias
con validación temporal, registrando Brier y probabilidades de premio por juego;
no aumentar artificialmente la probabilidad después de un acierto aislado.

### Evaluación arquitectónica y evolución a múltiples modelos

La arquitectura actual es adecuada como MVP reproducible: ingesta oficial,
normalización, SQLite, entrenamiento por `strategy/game`, evaluación y alerta.
Sus límites principales son el tamaño desigual del histórico, la ausencia de
validación temporal automatizada, una sola familia de modelo y la falta de una
calibración formal de probabilidades. En un sorteo diseñado para ser aleatorio,
un modelo más complejo no implica una ventaja predictiva; la métrica prioritaria
debe ser la calibración fuera de muestra, no la cantidad de aciertos aparentes.

La implementación actual ya incluye un ensamble supervisado por juego, ejecutado
en paralelo:

1. Generar características sin fuga temporal: calendario, rezagos, frecuencias
   con ventana móvil y datos de cada categoría de premio.
2. Entrenar modelos independientes: `RandomForest`, `ExtraTrees`,
   `HistGradientBoosting` y `MLPRegressor` opcional cuando el histórico alcanza
   el umbral configurado.
3. Evaluarlos con validación walk-forward. Los pesos del ensamble se obtienen
   solo de las predicciones históricas fuera de muestra, usando Brier o log loss;
   nunca del último acierto ni del correo enviado.
4. Calibrar las probabilidades con `CalibratedClassifierCV` o isotonic cuando
   exista suficiente histórico. Para cada candidato se conserva modelo, versión,
   ventana de entrenamiento, métrica y peso.
5. Promediar las distribuciones de probabilidad por juego, normalizarlas y
   publicar también la predicción de cada modelo, la predicción combinada y la
   incertidumbre. Si el ensamble no supera la línea base en walk-forward, se
   mantiene la línea base.

La ejecución paralela aísla fallos por modelo mediante `concurrent.futures`,
limita el paralelismo de cada estimador y guarda artefactos versionados. El
comando `backtest` ejecuta la validación walk-forward sobre el CSV histórico.
Todavía quedan como siguientes mejoras formales la calibración isotónica, una
tabla SQLite específica de métricas por modelo y una línea base uniforme
explícita. XGBoost/LightGBM, PyTorch o una red temporal profunda solo deben
incorporarse si el backtesting demuestra una mejora estable; con el histórico
actual una red profunda tiene alto riesgo de sobreajuste.


## <img src="https://media2.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif?cid=ecf05e47a0n3gi1bfqntqmob8g9aid1oyj2wr3ds3mg700bl&rid=giphy.gif" width="30px" style="vertical-align: middle;"> 🛠️ Technical Stack

<details>
<summary><strong>💻 Languages & Core Technologies</strong></summary>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://www.mysql.com/">
    <img src="https://img.shields.io/badge/SQL-CC2927?style=for-the-badge&logo=mysql&logoColor=white" alt="SQL"/>
  </a>
</p>

</details>

<details>
<summary><strong>🤖 Data Science & Machine Learning</strong></summary>

**Data Analysis & Processing**
<p align="center">
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  </a>
  <a href="https://scipy.org/">
    <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy"/>
  </a>
  <a href="https://openpyxl.readthedocs.io/">
    <img src="https://img.shields.io/badge/OpenPyXL-3C7C2A?style=for-the-badge&logo=python&logoColor=white" alt="OpenPyXL"/>
  </a>
</p>

**Machine Learning Libraries**
<p align="center">
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  </a>
  <a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  </a>
  <a href="https://keras.io/">
    <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>
</p>

**Data Visualization**
<p align="center">
  <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"/>
  </a>
  <a href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn"/>
  </a>
  <a href="https://plotly.com/">
    <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
  </a>
  <a href="https://powerbi.microsoft.com/">
    <img src="https://img.shields.io/badge/Power%20BI-F2CC8F?style=for-the-badge&logo=powerbi&logoColor=black" alt="Power BI"/>
  </a>
</p>

</details>

<details>
<summary><strong>🏗️ Data Engineering & Big Data</strong></summary>

**Distributed Computing & Processing**
<p align="center">
  <a href="https://spark.apache.org/">
    <img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="Apache Spark"/>
  </a>
  <a href="https://spark.apache.org/docs/latest/api/python/">
    <img src="https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="PySpark"/>
  </a>
  <a href="https://hadoop.apache.org/">
    <img src="https://img.shields.io/badge/Hadoop-66CCFF?style=for-the-badge&logo=apache&logoColor=black" alt="Hadoop"/>
  </a>
</p>

**Orchestration & Infrastructure**
<p align="center">
  <a href="https://airflow.apache.org/">
    <img src="https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white" alt="Apache Airflow"/>
  </a>
  <a href="https://impala.apache.org/">
    <img src="https://img.shields.io/badge/Apache%20Impala-FF6F00?style=for-the-badge&logo=apache&logoColor=white" alt="Apache Impala"/>
  </a>
  <a href="https://www.cloudera.com/">
    <img src="https://img.shields.io/badge/Cloudera-0066CC?style=for-the-badge&logo=cloudera&logoColor=white" alt="Cloudera"/>
  </a>
  <a href="https://www.sqlalchemy.org/">
    <img src="https://img.shields.io/badge/SQLAlchemy-CC2927?style=for-the-badge&logo=sqlalchemy&logoColor=white" alt="SQLAlchemy"/>
  </a>
</p>

</details>

<details>
<summary><strong>🗄️ Databases & Data Storage</strong></summary>

**Relational Databases**
<p align="center">
  <a href="https://www.mysql.com/">
    <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white" alt="MySQL"/>
  </a>
  <a href="https://www.postgresql.org/">
    <img src="https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  </a>
</p>

**NoSQL Databases**
<p align="center">
  <a href="https://www.mongodb.com/">
    <img src="https://img.shields.io/badge/MongoDB-13AA52?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/>
  </a>
  <a href="https://www.sqlite.org/">
    <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
  </a>
</p>

</details>

<details>
<summary><strong>🕸️ Data Collection & Web Scraping</strong></summary>

<p align="center">
  <a href="https://selenium.dev/">
    <img src="https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white" alt="Selenium"/>
  </a>
  <a href="https://www.crummy.com/software/BeautifulSoup/">
    <img src="https://img.shields.io/badge/BeautifulSoup-336666?style=for-the-badge&logo=python&logoColor=white" alt="BeautifulSoup"/>
  </a>
</p>

</details>

<details>
<summary><strong>🛠️ Development Tools & Environment</strong></summary>

**Version Control**
<p align="center">
  <a href="https://git-scm.com/">
    <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"/>
  </a>
  <a href="https://github.com/">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

**IDEs & Notebooks**
<p align="center">
  <a href="https://code.visualstudio.com/">
    <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" alt="VS Code"/>
  </a>
  <a href="https://jupyter.org/">
    <img src="https://img.shields.io/badge/Jupyter-F37726?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  </a>
  <a href="https://www.jetbrains.com/pycharm/">
    <img src="https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white" alt="PyCharm"/>
  </a>
</p>

**OS & Platforms**
<p align="center">
  <a href="https://ubuntu.com/">
    <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Ubuntu"/>
  </a>
</p>

</details>

<!-- ## Dashboards

| **Dashboard 1** | **Dashboard 2** |
|---|---|
| [![Dashboard 1](informe_1.png)](https://app.powerbi.com/view?r=eyJrIjoiN2RmZmE5YjUtMmQzMy00YjhiLWE1NmEtZDg5OTBjNWU2OTA1IiwidCI6ImNhM2YxZDZiLWZkMWYtNDBiMS1iNDFhLTQ4OGI5ODBlOWY3ZiIsImMiOjR9&pageName=ReportSectioncb1ce34056930c0b40d2) | [![Dashboard 2](informe_2.png)](https://app.powerbi.com/view?r=eyJrIjoiN2RmZmE5YjUtMmQzMy00YjhiLWE1NmEtZDg5OTBjNWU2OTA1IiwidCI6ImNhM2YxZDZiLWZkMWYtNDBiMS1iNDFhLTQ4OGI5ODBlOWY3ZiIsImMiOjR9&pageName=ReportSectioncb1ce34056930c0b40d2) |
| _Descripción breve del dashboard 1._ | _Descripción breve del dashboard 2._ | -->

## 📊 GitHub Stats
<br>
<p align="center">
  <a href="https://github.com/JairTrujillo-R/beautiful-github-homepage"><picture><img align="" height='150px' src="https://beautiful-github-homepage.vercel.app/api?username=JairTrujillo-R&hide_title=true&show_icons=true&theme=gotham&include_all_commits=true" alt="GitHub Stats" /></picture></a>
  <a href="https://github.com/JairTrujillo-R/beautiful-github-homepage"><picture><img align="" height='150px' src="https://beautiful-github-homepage.vercel.app/api/top-langs/?username=JairTrujillo-R&hide_title=false&layout=compact&theme=gotham&count_private=true", alt="Top Languajes"/></picture></a>
</p>

## 🔗 Contact Me

<div align="center">
<a href="mailto:coremprejair@gmail.com"><img alt="Gmail" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
<a href="https://www.linkedin.com/in/jairtrujillo"><img alt="LinkedIn" src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
</div>

---

<div align="center">
<!-- ![Profile views](https://komarev.com/ghpvc/?username=JairTrujillo-R&color=blueviolet) -->

![Profile views](https://komarev.com/ghpvc/?username=JairTrujillo-R&label=PROFILE%20VIEWS&color=007BFF&labelColor=555555&style=flat-square)
<a href="https://github.com/JairTrujillo-R?tab=followers">
  <img height="24" alt="GitHub followers" src="https://img.shields.io/github/followers/JairTrujillo-R?style=for-the-badge&logo=github&logoColor=white"/>
</a>
<a href="https://github.com/JairTrujillo-R?tab=stars">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/JairTrujillo-R?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<br>

**Made with ❤️ by [Jair Trujillo](https://github.com/JairTrujillo-R)**

</div>