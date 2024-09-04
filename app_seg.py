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
        row_segments = []
        for j in range(3):
            left = j * segment_width
            upper = i * segment_height
            right = (j + 1) * segment_width
            lower = (i + 1) * segment_height
            segment = image.crop((left, upper, right, lower))
            row_segments.append(segment)
        segments.append(row_segments)
    
    return segments

# Fonction pour prédire les probabilités de poussière sur chaque segment
def predict_dust_probability(model, image):
    segments = segment_image(image)
    
    # Initialiser une matrice 3x3 pour les labels
    labels = np.empty((3, 3), dtype=object)
    
    for i in range(3):
        for j in range(3):
            segment = segments[i][j]
            segment = segment.convert('RGB')  # Convertir en RGB pour assurer 3 canaux
            segment = segment.resize((150, 150))  # Redimensionner si nécessaire
            
            segment_array = np.array(segment) / 255.0  # Normaliser l'image
            segment_array = np.expand_dims(segment_array, axis=0)  # Ajouter une dimension pour le batch
            
            # Prédire la probabilité de poussière pour cette section
            prob = model.predict(segment_array)[0][0]
            
            # Assigner un label basé sur le seuil de 0,5
            labels[i, j] = 'with_dust' if prob >= 0.5 else 'without_dust'
    
    return labels

# Fonction pour calculer la proportion de segments avec de la poussière
def calculate_dust_proportion(labels):
    return np.sum(labels == 'with_dust') / labels.size

# Titre de l'application
st.title("Évolution de la détection de Dust avec Deep Learning")

# Uploader pour charger plusieurs images
uploaded_files = st.file_uploader("Choisissez des images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    dust_proportions = []

    for uploaded_file in uploaded_files:
        try:
            # Ouvrir l'image avec PIL
            image = Image.open(uploaded_file)

            # Afficher l'image téléchargée
            st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)

            # Faire une prédiction en utilisant la fonction segmentée
            dust_labels = predict_dust_probability(model, image)
            
            # Calculer la proportion de segments avec de la poussière
            dust_proportion = calculate_dust_proportion(dust_labels)
            dust_proportions.append(dust_proportion)
            
            # Afficher le résultat sous forme de matrice 3x3
            st.write("Matrice 3x3 des labels 'with_dust' ou 'without_dust':")
            st.write(dust_labels)
            
            st.write(f"Proportion de poussière dans cette image : {dust_proportion:.2%}")
            
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image {uploaded_file.name} : {e}")
    
    # Afficher le graphe de l'évolution de la proportion de poussière
    st.write("Évolution de la proportion de poussière dans les images téléchargées :")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(dust_proportions) + 1), dust_proportions, marker='o')
    plt.xlabel("Images")
    plt.ylabel("Proportion de poussière (%)")
    plt.title("Évolution de la proportion de poussière")
    plt.grid(True)
    st.pyplot(plt)

else:
    st.write("Veuillez télécharger des images pour obtenir une prédiction.")
