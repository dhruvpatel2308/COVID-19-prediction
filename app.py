import streamlit as st
import requests
import json
from io import BytesIO

st.title('COVID Prediction')

image_file = st.file_uploader("Upload Image")

if image_file is not None:

    image = image_file.getvalue()

    response = requests.post(
        "http://127.0.0.1:3000/prediction",
        files = {
            "image": BytesIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Patient is {label['class']}")