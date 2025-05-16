# Análisis estilométrico automatizado de textos latinos

Este proyecto proporciona una interfaz interactiva basada en Streamlit para realizar análisis estilométrico de textos latinos. Su principal objetivo es facilitar la atribución de autoría mediante técnicas de aprendizaje automático, con un enfoque en la comparación de patrones estilísticos a partir de n-gramas de caracteres.

## Estructura esperada de datos

El sistema trabaja con archivos `.zip` que contienen textos en formato `.txt`. Cada archivo debe estar nombrado comenzando por el nombre del autor, seguido de un guion bajo y el título del texto. Ejemplo:

```
Tacito_Anales16.txt
Caesar_BelloCivili.txt
```

Se requieren dos conjuntos de datos:

* Un conjunto de **entrenamiento**, que contiene textos con autor conocido.
* Un conjunto de **clasificación**, con textos cuya autoría se desea predecir.

## Proceso de análisis

1. **Carga y preparación de datos:** Los textos se cargan desde el archivo ZIP y se extraen en carpetas temporales.

2. **Extracción de características estilísticas:** Se utiliza `TfidfVectorizer` con `analyzer='char'` para representar los textos mediante n-gramas de caracteres. El usuario puede seleccionar el rango de `n` mediante la barra lateral de configuración.

3. **Reducción de dimensionalidad:** Se aplica `TruncatedSVD` para reducir la dimensionalidad del espacio vectorial a 50 componentes, permitiendo un procesamiento más eficiente sin pérdida significativa de información.

4. **División del conjunto de entrenamiento:** El 80% de los textos etiquetados se utilizan para entrenar el modelo, y el 20% restante se reserva para pruebas. La partición se estratifica para preservar la proporción de autores.

5. **Modelado:** Se entrena un clasificador `NearestCentroid`, que asigna nuevos textos al autor cuyo centroide estilístico se encuentre más próximo en el espacio reducido.

6. **Evaluación del modelo:** Se muestra la matriz de confusión y el porcentaje de acierto sobre el conjunto de test.

7. **Clasificación de nuevos textos:** Al cargar un segundo archivo ZIP, el sistema calcula las distancias de los textos nuevos a los centroides de autor y estima probabilidades de autoría mediante una softmax inversa.

## Resultados y visualizaciones

* **Matriz de confusión** con los resultados del conjunto de test.
* **Tabla de distancias** de los nuevos textos a los centroides de autor.
* **Tabla de probabilidades de autoría**, expresadas en porcentaje.
* **Gráficos de barras** para cada texto, representando la distribución de probabilidades por autor.

