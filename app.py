import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to predict its disease.")

REPO_ID = "NoorFatima10/plant-disease-detection"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="plant_disease_model.keras"
    )
    labels_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="class_indices.json"
    )

    model = tf.keras.models.load_model(model_path)

    with open(labels_path, "r") as f:
        class_indices = json.load(f)

    labels = {v: k for k, v in class_indices.items()}
    return model, labels

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing image..."):
            model, labels = load_model()

            image = image.resize((128, 128))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array)
            predicted_index = int(np.argmax(prediction))
            disease = labels[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.success(f"Predicted Disease: {disease}")
        st.write(f"Confidence: {confidence:.2f}%")
