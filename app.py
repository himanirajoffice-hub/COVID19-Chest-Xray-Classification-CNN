
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-ray Classifier",
    page_icon="🩻",
    layout="centered"
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():

    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(128, 128, 3)
    )

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Lambda(
            lambda x: preprocess_input(x * 255.0)
        ),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            3,
            activation="softmax"
        )
    ])

    model.load_weights("best_vgg16.weights.h5")

    return model


model = load_model()


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]


# --------------------------------------------------
# APP TITLE
# --------------------------------------------------
st.title("🩻 COVID-19 Chest X-ray Classifier")

st.write(
    """
    This deep learning application classifies a chest X-ray
    into one of three categories:

    **COVID-19 • Normal • Viral Pneumonia**
    """
)

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-ray")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )


    # ----------------------------------------------
    # PREPROCESS IMAGE
    # ----------------------------------------------
    resized_image = image.resize((128, 128))

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    if st.button("Analyze X-ray"):

        with st.spinner("Analyzing image..."):

            prediction = model.predict(
                image_array,
                verbose=0
            )[0]

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[
            predicted_index
        ]

        confidence = prediction[
            predicted_index
        ] * 100


        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------
        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            class_names,
            prediction
        ):

            st.write(
                f"**{class_name}:** "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.divider()

st.warning(
    """
    Educational Use Only:
    This application is a machine learning demonstration
    developed for an academic project. It is not a medical
    diagnostic tool and should not be used for clinical
    diagnosis or medical decision-making.
    """
)
