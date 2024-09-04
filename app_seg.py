import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le modèle TensorFlow .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_seg.h5")

model = load_model()

# Fonction pour segmenter l'image en 9 parties égales
def segment_image(image):
    """Segmenter une image en 9 sections égales et retourner une liste avec chaque section."""
    width, height = image.size
    segment_width = width // 3
    segment_height = height // 3
    segments = []
    
    for i in range(3):
        for j in range(3):
            left = j * segment_width
            upper = i * segment_height
            right = (j + 1) * segment_width
            lower = (i + 1) * segment_height
            segment = image.crop((left, upper, right, lower))
            segments.append(segment)
    
    return segments

# Fonction pour faire des prédictions sur les segments
def predict_dust_probability(model, segments):
    """Appliquer le modèle sur chaque section d'une image et retourner une liste avec les labels 'with_dust' ou 'without_dust'."""
    labels = []
    
    for segment in segments:
        segment = segment.resize((150, 150))
        segment_array = np.array(segment) / 255.0
        segment_array = segment_array[:, :, :3]  # S'assurer que l'image est en RGB
        segment_array = np.expand_dims(segment_array, axis=0)
        
        prob = model.predict(segment_array)[0][0]
        if prob >= 0.5:
            labels.append('with_dust')
        else:
            labels.append('without_dust')
    
    return labels

# Titre de l'application
st.title("Détection de Dust avec Deep Learning")

# Uploader pour charger une image
uploaded_files = st.file_uploader("Choisissez des images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Si des fichiers sont téléchargés
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Ouvrir l'image avec PIL
            image = Image.open(uploaded_file)

            # Afficher l'image téléchargée
            st.image(image, caption=f"Image {uploaded_file.name} chargée avec succès", use_column_width=True)

            # Segmenter l'image
            segments = segment_image(image)

            # Faire des prédictions
            labels = predict_dust_probability(model, segments)

            # Calculer la proportion de dust par segment
            dust_proportions = [1 if label == 'with_dust' else 0 for label in labels]

            # Tracer un graphe par portion
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, 10), dust_proportions)
            plt.xlabel('Segment')
            plt.ylabel('Présence de Dust (1 = avec dust, 0 = sans dust)')
            plt.title(f"Évolution de la détection de Dust pour l'image {uploaded_file.name}")
            plt.ylim(0, 1)
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image {uploaded_file.name} : {e}")
        finally:
            plt.close('all')
else:
    st.write("Veuillez télécharger une ou plusieurs images pour obtenir des prédictions.")
