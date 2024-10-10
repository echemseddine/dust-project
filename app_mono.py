import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Charger le modèle TensorFlow .h5
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_classification_dust.h5")

model = load_model()

# Fonction pour prétraiter l'image
def preprocess_image(image):
    try:
        # Afficher la taille originale de l'image
        original_size = image.size
        st.write(f"Taille originale de l'image : {original_size}")

        # Convertir l'image en RGB (supprimer le canal alpha si présent)
        image = image.convert("RGB")
        
        # Redimensionner l'image à la taille d'entrée du modèle
        image = image.resize((498, 498))  # Redimensionner à 498x498
        image = np.array(image)  # Convertir l'image en tableau numpy
        image = image / 255.0  # Normaliser l'image
        image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
        
        # Afficher les dimensions pour déboguer
        st.write(f"Dimensions après prétraitement : {image.shape}")

        return image
    except Exception as e:
        st.error(f"Erreur lors du prétraitement de l'image : {e}")
        return None

# Fonction pour faire une prédiction
def make_prediction(image, model):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return None
    try:
        # Afficher les dimensions de l'image traitée pour le débogage
        st.write(f"Dimensions de l'image traitée : {processed_image.shape}")

        # Faire une prédiction
        prediction = model.predict(processed_image)
        return prediction[0][0]  # Supposant que le modèle retourne une seule prédiction
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None

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

        # Faire une prédiction
        prediction = make_prediction(image, model)

        # Afficher le résultat
        if prediction is not None:
            st.write(f"Valeur de la prédiction : {prediction:.4f}")
            if prediction > 0.5:
                st.write("Résultat : Dust détecté.")
            else:
                st.write("Résultat : Pas de dust détecté.")
        else:
            st.write("Erreur lors de la prédiction.")
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
else:
    st.write("Veuillez télécharger une image pour obtenir une prédiction.")
