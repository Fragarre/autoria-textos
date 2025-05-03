# Clasificación y Visualización de Textos en Prosa Latina
Este código es una adptación a un front-end streamlit del proyecto base.

## Referencias y créditos

Este proyecto se inspira en el trabajo de Benjamin Nagy sobre estilometría aplicada a textos latinos. Agradezco especialmente su artículo:

> Nagy, Benjamin. *Some stylometric remarks on Ovid’s Heroides and the Epistula Sapphus*. Digital Scholarship in the Humanities, 2023. [https://doi.org/10.1093/llc/fqac098](https://doi.org/10.1093/llc/fqac098)

El repositorio asociado al artículo original está licenciado bajo **CC-BY 4.0**, lo que permite su reutilización con atribución adecuada. Parte del enfoque metodológico y elementos del código fueron adaptados de dicho trabajo con respeto a esta licencia. (https://github.com/bnagy/heroides-paper)


## Explicación de las capacidades y funcionamiento del programa

Este programa está diseñado para realizar un análisis estilométrico de textos.

## 1. Vectorización de textos con TF-IDF:

Los textos se convierten en vectores numéricos utilizando el modelo de **TF-IDF** (*Term Frequency–Inverse Document Frequency*), que pondera la frecuencia de aparición de fragmentos (o *tokens*) dentro de cada documento, considerando su rareza en el conjunto global.

- Se utiliza un `TfidfVectorizer` de `scikit-learn` con `analyzer='char'`, lo que significa que los textos se analizan como secuencias de **caracteres**, no de palabras.
- Se aplican **n-grams de caracteres**, que son cadenas consecutivas de `n` letras. Por ejemplo, el 3-gram de la palabra `"texto"` generaría: `"tex"`, `"ext"`, `"xto"`.
- El rango de n-gramas (`ngram_min`, `ngram_max`) es configurable desde la interfaz. Por defecto, se usa de 2 a 4, es decir, 2-gramas, 3-gramas y 4-gramas.

## 2. Reducción de dimensión: TruncatedSVD
Como el espacio generado por los n-grams puede tener decenas de miles de dimensiones, se aplica Truncated Singular Value Decomposition (TruncatedSVD), también conocido como LSA (Latent Semantic Analysis).

Esto reduce la complejidad del modelo sin perder las características más relevantes del estilo de escritura.

## 3. Clasificación: Nearest Centroid
El clasificador entrenado es un modelo simple y eficiente: Nearest Centroid, que calcula el centroide (punto medio) de cada clase y predice la clase del nuevo texto según el centroide más cercano.

## 4.Matriz de confusión:

Tras entrenar el modelo, el programa calcula una matriz de confusión, que compara las predicciones del clasificador con las etiquetas reales de los textos. Esta matriz muestra cuántos textos fueron correctamente clasificados y cuántos fueron mal clasificados, lo que permite identificar patrones de errores y mejorar el modelo.

## 5.Visualizaciones: t-SNE y UMAP:

El programa incluye dos métodos de reducción de dimensionalidad para crear representaciones visuales de los datos: t-SNE y UMAP. Estas técnicas permiten proyectar los datos de alta dimensión (vectores de texto) en un espacio bidimensional, lo que facilita la visualización de cómo se agrupan los textos en función de su autoría. Estas visualizaciones ayudan a comprender cómo el clasificador agrupa los textos y cómo se distribuyen los diferentes autores en el espacio de características.

# ¿Qué es el clasificador Nearest Centroid?

**Nearest Centroid** es un algoritmo de clasificación supervisado, simple y eficiente, especialmente útil cuando las clases tienen una distribución bien separada en el espacio vectorial. Este modelo se basa en calcular el centro (o promedio) de los vectores de cada clase, y luego asigna nuevas muestras a la clase cuyo centroide esté más cercano.

##  ¿Cómo funciona?

Para cada clase (por ejemplo, un autor), el algoritmo calcula el **centroide**, que es el promedio de todos los vectores que pertenecen a esa clase: μ_c = (1 / N_c) * Σ x_i.

donde:
- `μ_c` es el centroide de la clase `c`.
- `N_c` es el número de textos de esa clase.
- `x_i` son los vectores TF-IDF reducidos mediante SVD correspondientes a los textos.
