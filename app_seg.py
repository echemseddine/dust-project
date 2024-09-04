import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

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

        # Étiquettes directionnelles dans l'ordre spécifié
        directions = [
            ['NW', 'N', 'NE'],
            ['W', 'C', 'E'],
            ['SW', 'S', 'SE']
        ]

        # Générer le HTML pour afficher le tableau avec des arrière-plans directionnels
        html = '<table style="border-collapse: collapse; width: 100%;">'
        for i in range(3):
            html += '<tr>'
            for j in range(3):
                # Définir le style pour la cellule en fonction de la prédiction
                bg_color = '#ff00ff' if dust_probabilities[i, j] == 'with_dust' else '#ffffff'
                cell_style = (
                    f'border: 1px solid black; padding: 10px; text-align: center; '
                    f'background-color: {bg_color}; height: 100px; width: 100px;'
                )
                # Construire le contenu de la cellule avec le label directionnel et la prédiction
                direction = directions[i][j]
                html += f'<td style="{cell_style}">'
                html += f'<div style="position: relative; width: 100%; height: 100%;">'
                html += f'<span style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; '
                html += f'display: flex; align-items: center; justify-content: center; '
                html += f'color: gray; font-size: 14px;">{direction}</span>'
                html += f'<span style="position: relative; z-index: 1; display: flex; align-items: center; '
                html += f'justify-content: center; height: 100%;"></span>'
                html += '</div></td>'
            html += '</tr>'
        html += '</table>'

        # Ajouter une légende en bas
        legend_html = '''
        <div style="margin-top: 20px;">
            <p><span style="background-color: #ff00ff; padding: 5px; color: white;">Poussière détectée</span></p>
            <p><span style="background-color: #ffffff; padding: 5px;">Pas de poussière détectée</span></p>
        </div>
        '''
        html += legend_html

        # Afficher le tableau HTML
        st.markdown(html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
else:
    st.write("Veuillez télécharger une image pour obtenir une prédiction.")
