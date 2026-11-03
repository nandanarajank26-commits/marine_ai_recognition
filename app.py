import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Marine Animal AI")

st.title("🐢 Marine Animal Recognition")
st.write("Upload an image to identify the marine animal.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_marine_model.keras")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a marine animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        classes = ["Seahorse", "Turtle_Tortoise"]
        predicted_class = classes[int(prediction[0][0] >= 0.5)]

        st.success(f"Prediction: {predicted_class}")
