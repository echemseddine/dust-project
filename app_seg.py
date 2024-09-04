import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf

# Charger le modèle TensorFlow .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_seg.h5")

model = load_model()

# Fonction pour segmenter l'image en 9 parties égales
def segment_image(image):
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
    
    labels = np.empty((3, 3), dtype=object)
    
    for i in range(3):
        for j in range(3):
            segment = segments[i][j]
            segment = segment.convert('RGB')
            segment = segment.resize((150, 150))
            segment_array = np.array(segment) / 255.0
            segment_array = np.expand_dims(segment_array, axis=0)
            prob = model.predict(segment_array)[0][0]
            labels[i, j] = 'with_dust' if prob >= 0.5 else 'without_dust'
    
    return labels

# Fonction pour dessiner les lignes et ajouter des étiquettes directionnelles
def draw_divisions_and_labels(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    segment_width = width // 3
    segment_height = height // 3

    # Couleur des lignes de division
    line_color = (255, 0, 0)  # Rouge pour les lignes de séparation
    label_color = (0, 0, 0)  # Noir pour les étiquettes directionnelles
    font_size = 20

    # Dessiner les lignes de séparation
    for i in range(1, 3):
        # Lignes verticales
        draw.line([(i * segment_width, 0), (i * segment_width, height)], fill=line_color, width=2)
        # Lignes horizontales
        draw.line([(0, i * segment_height), (width, i * segment_height)], fill=line_color, width=2)
    
    # Ajouter les étiquettes directionnelles
    directions = [
        ['NW', 'N', 'NE'],
        ['W', 'C', 'E'],
        ['SW', 'S', 'SE']
    ]
    for i in range(3):
        for j in range(3):
            direction = directions[i][j]
            x = j * segment_width + segment_width // 2
            y = i * segment_height + segment_height // 2
            draw.text((x, y), direction, fill=label_color, anchor="mm", font=None)

    return image

# Titre de l'application
st.title("Détection de Dust avec Deep Learning")

# Uploader pour charger une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Ouvrir l'image avec PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée avec succès", use_column_width=True)
        
        # Dessiner les lignes et ajouter les étiquettes directionnelles
        image_with_labels = draw_divisions_and_labels(image.copy())

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
                # Construire le contenu de la cellule avec le label directionnel
                direction = directions[i][j]
                html += f'<td style="{cell_style}">'
                html += f'<div style="position: relative; width: 100%; height: 100%;">'
                html += f'<span style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; '
                html += f'display: flex; align-items: center; justify-content: center; '
                html += f'color: black; font-size: 14px;">{direction}</span>'
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

        # Afficher l'image avec les étiquettes directionnelles
        st.image(image_with_labels, caption="Image avec divisions et étiquettes", use_column_width=True)

        # Afficher le tableau HTML
        st.markdown(html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
else:
    st.write("Veuillez télécharger une image pour obtenir une prédiction.")
