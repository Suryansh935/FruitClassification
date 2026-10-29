
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make the backend directory importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from predict import load_models, transform, DEVICE
import torch


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("Fruit Freshness Classifier")
st.write(
    "Upload an image of an apple, banana, or orange. "
    "Our two CNN models will identify the fruit and predict its condition."
)

st.info(
    "Model 1 identifies the fruit type. "
    "Model 2 classifies the fruit as fresh or rotten."
)


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def get_models():
    return load_models()


# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Classify Fruit", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                fruit_model, freshness_model, fruit_classes, freshness_classes = (
                    get_models()
                )

                image_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.inference_mode():
                    fruit_output = fruit_model(image_tensor)
                    freshness_output = freshness_model(image_tensor)

                    fruit_probs = torch.softmax(fruit_output, dim=1)[0]
                    freshness_probs = torch.softmax(
                        freshness_output, dim=1
                    )[0]

                    fruit_idx = fruit_probs.argmax().item()
                    freshness_idx = freshness_probs.argmax().item()

                    fruit_name = fruit_classes[fruit_idx].title()
                    freshness_name = freshness_classes[
                        freshness_idx
                    ].title()

                    fruit_confidence = fruit_probs[fruit_idx].item() * 100
                    freshness_confidence = (
                        freshness_probs[freshness_idx].item() * 100
                    )

            st.subheader("Classification Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Fruit Type", fruit_name)
                st.metric(
                    f"{freshness_name} Confidence",
                    f"{freshness_confidence:.2f}%"
                )

            with col2:
                st.metric("Condition", freshness_name)
                st.metric(
                    f"{freshness_name} Confidence",
                    f"{freshness_confidence:.2f}%"
                )

            st.caption(
                "Confidence scores represent the models' predicted-class "
                "probabilities, not guaranteed correctness."
            )

        except (FileNotFoundError, KeyError, RuntimeError, OSError) as error:
            st.error(f"Could not classify the image: {error}")

else:
    st.caption("Upload an image to get started.")

st.divider()
st.caption("Two custom CNN models trained from scratch using PyTorch.")
