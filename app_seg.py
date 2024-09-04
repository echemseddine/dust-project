import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

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

# Titre de l'application
st.title("Détection de Dust avec Deep Learning")

# Uploader pour charger une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# Si un fichier est téléchargé
if uploaded_file is not None:
    try:
        # Ouvrir l'image avec PIL
        image = Image.open(uploaded_file)

        # Afficher l'image téléchargée
        st.image(image, caption="Image chargée avec succès", use_column_width=True)

        # Faire une prédiction en utilisant la fonction segmentée
        dust_probabilities = predict_dust_probability(model, image)

        # Afficher le résultat sous forme de matrice 3x3
        st.write("Matrice 3x3 des labels 'with_dust' ou 'without_dust':")
        st.write(dust_probabilities)
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
else:
    st.write("Veuillez télécharger une image pour obtenir une prédiction.")
