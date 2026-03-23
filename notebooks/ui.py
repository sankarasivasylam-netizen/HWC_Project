import streamlit as st
import requests

st.title("Image Classifier")

file = st.file_uploader("Upload Image")

if file:
    st.image(file, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": file.getvalue()}
        )

        data = res.json()

        st.success(f"Prediction: {data['predicted_label']}")
        st.write(f"Confidence: {round(data['confidence']*100,2)} %")