import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
# import umap
from umap import UMAP
import plotly.express as px
import pandas as pd

# --- Función para extraer el archivo ZIP --- #
def unzip_data(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('./data/')
    st.success("¡Datos extraídos correctamente!. (Espera mientras el estado sea RUNNING...)")

# --- Función para cargar los textos y generar el modelo --- #
def load_and_train_model(ngram_min, ngram_max):
    corpus_path = './data/'
    texts = []
    labels = []
    filenames = []

    for author_folder in os.listdir(corpus_path):
        author_path = os.path.join(corpus_path, author_folder)
        if os.path.isdir(author_path):
            for filename in os.listdir(author_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(author_path, filename), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(author_folder)
                        filenames.append(f"{author_folder}/{filename[:-4]}")  # Nombre autor/archivo

    # Vectorización de textos
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max)) # Cambia aquí
    X = vectorizer.fit_transform(texts)

    # Reducción de dimensión con SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Clasificación con NearestCentroid
    clf = NearestCentroid()
    clf.fit(X_reduced, labels)
    
    return clf, vectorizer, svd, texts, labels, filenames, X_reduced

# --- Función para identificar los errores en la matriz de confusión --- #
def identify_confusion_errors(labels, y_pred, filenames, clf):
    # Identificar las posiciones fuera de la diagonal
    errors = []
    cm = confusion_matrix(labels, y_pred, labels=clf.classes_)

    # Recorrer la matriz de confusión y buscar los errores
    for i in range(len(clf.classes_)):
        for j in range(len(clf.classes_)):
            if i != j and cm[i, j] > 0:
                # Encontrar los índices de las obras que no están en la diagonal
                misclassified_files = [filenames[idx] for idx, label in enumerate(labels) 
                                       if label == clf.classes_[i] and y_pred[idx] == clf.classes_[j]]
                errors.append({
                    'true_label': clf.classes_[i],
                    'predicted_label': clf.classes_[j],
                    'misclassified_files': misclassified_files
                })
    return errors

# --- Interfaz de usuario en Streamlit --- #
st.title("Análisis estilométrico. Textos latinos")
st.write("Sube un archivo .zip con los datos de los autores para entrenar el modelo.")
st.sidebar.markdown("""
### Instrucciones

Para la descarga de los ficheros, debes tener un fichero `data.zip` que contendrá los autores y obras.  
Ejemplo de estructura:

/Ciceron                           
/Plinio                 


Selecciona conjuntamente las carpetas Ciceron y Plinio y zipealas. Al archivo resultante llamalo data.zip
Este data.zip es el fichero a subir.
                    
""")
st.sidebar.markdown("### Configuración de n-gramas")
ngram_min = st.sidebar.number_input("n-grama mínimo", min_value=1, max_value=10, value=2)
ngram_max = st.sidebar.number_input("n-grama máximo", min_value=1, max_value=10, value=4)

uploaded_zip = st.file_uploader("Sube un archivo .zip", type=["zip"])

if uploaded_zip is not None:
    # Extraer el contenido del archivo ZIP
    unzip_data(uploaded_zip)

    # Cargar y entrenar el modelo
    clf, vectorizer, svd, texts, labels, filenames, X_reduced = load_and_train_model(ngram_min, ngram_max)

    # --- Visualización de la matriz de confusión --- #
    st.write("Generando la matriz de confusión...")
    y_pred = clf.predict(X_reduced)
    cm = confusion_matrix(labels, y_pred, labels=clf.classes_)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cm_display.plot(cmap=plt.cm.Blues)
    st.pyplot(plt.gcf(), use_container_width=True)

    # Generar visualizaciones (t-SNE y UMAP)
    st.write("Generando visualizaciones...")
    
    # --- Reducción con t-SNE y UMAP --- #
    tsne = TSNE(n_components=2, random_state=42, perplexity=30) 
    X_tsne = tsne.fit_transform(X_reduced)

    # reducer = umap.UMAP(n_components=2, random_state=42)
    reducer = UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_reduced)

    # Crear las visualizaciones t-SNE y UMAP
    tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
    tsne_df['author'] = labels
    tsne_df['filename'] = filenames
    fig_tsne = px.scatter(tsne_df, x='x', y='y', color='author', hover_data=['filename'], 
                          title="Visualización t-SNE", width=900, height=600)
    fig_tsne.update_traces(marker=dict(size=12))  # Ajusta el tamaño de los puntos
    st.plotly_chart(fig_tsne)
    # fig_tsne.write_html("visualizaciones/tsne.html")

    umap_df = pd.DataFrame(X_umap, columns=['x', 'y'])
    umap_df['author'] = labels
    umap_df['filename'] = filenames
    fig_umap = px.scatter(umap_df, x='x', y='y', color='author', hover_data=['filename'], 
                          title="Visualización UMAP", width=900, height=600)
    fig_umap.update_traces(marker=dict(size=12))  # Ajusta el tamaño de los puntos
    st.plotly_chart(fig_umap)
    # fig_umap.write_html("visualizaciones/umap.html")

# Descargar Readme
import base64

def get_readme_download_link(readme_path="README.md"):
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="README.md" target="_blank">📖 Descargar el fichero README.md</a>'
    return href

st.markdown(get_readme_download_link(), unsafe_allow_html=True)
