# =========================================================
# COMPUTER VISION - OBJECT DETECTION + COUNTING + EMOTION
# =========================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Vision - YOLO & Emotion Detection",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 AI Computer Vision System")

st.write(
    "This AI application detects and counts objects using YOLO "
    "and analyzes human emotions using DeepFace."
)

st.markdown("---")


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    model = YOLO("yolo11n.pt")

    return model


model = load_yolo_model()


# =========================================================
# SAMPLE IMAGE
# =========================================================

sample_image_url = (
    "https://ultralytics.com/images/bus.jpg"
)


# =========================================================
# FUNCTIONS
# =========================================================

def detect_objects(image):

    with st.spinner("🤖 YOLO is detecting objects..."):

        results = model.predict(
            source=image,
            conf=0.25
        )

    return results


def get_detected_objects(results):

    detected_objects = []

    if results[0].boxes is not None:

        for cls in results[0].boxes.cls:

            class_id = int(cls)

            class_name = model.names[class_id]

            detected_objects.append(
                class_name
            )

    return detected_objects


def show_object_results(results, detected_objects):

    # =====================================================
    # DETECTION IMAGE
    # =====================================================

    result_image = results[0].plot()

    st.subheader(
        "🎯 Object Detection Result"
    )

    st.image(
        result_image,
        caption="Detected Objects",
        use_container_width=True
    )


    # =====================================================
    # OBJECT COUNTING
    # =====================================================

    if len(detected_objects) > 0:

        object_counts = pd.Series(
            detected_objects
        ).value_counts()

        object_df = (
            object_counts
            .reset_index()
        )

        object_df.columns = [
            "Object",
            "Count"
        ]


        # =================================================
        # TABLE
        # =================================================

        st.subheader(
            "📊 Detected Objects Count"
        )

        st.dataframe(
            object_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # TOTAL OBJECTS
        # =================================================

        total_objects = len(
            detected_objects
        )

        st.metric(
            "🔢 Total Objects Detected",
            total_objects
        )


        # =================================================
        # BAR CHART
        # =================================================

        st.subheader(
            "📈 Object Count Visualization"
        )

        st.bar_chart(
            object_df.set_index(
                "Object"
            )["Count"]
        )

    else:

        st.warning(
            "⚠️ No objects were detected."
        )


def detect_emotions(image):

    st.subheader(
        "😊 Emotion Detection"
    )

    with st.spinner(
        "🧠 AI is analyzing facial emotions..."
    ):

        try:

            # Convert PIL image to NumPy
            image_array = np.array(image)

            # Analyze emotions
            emotion_results = DeepFace.analyze(
                img_path=image_array,
                actions=["emotion"],
                enforce_detection=False
            )

            # Make sure result is a list
            if not isinstance(
                emotion_results,
                list
            ):

                emotion_results = [
                    emotion_results
                ]


            emotions = []

            for result in emotion_results:

                dominant_emotion = result.get(
                    "dominant_emotion",
                    "Unknown"
                )

                emotions.append(
                    dominant_emotion
                )


            # =================================================
            # DISPLAY EMOTIONS
            # =================================================

            if len(emotions) > 0:

                emotion_counts = pd.Series(
                    emotions
                ).value_counts()


                emotion_df = (
                    emotion_counts
                    .reset_index()
                )


                emotion_df.columns = [
                    "Emotion",
                    "Count"
                ]


                st.success(
                    f"😊 Detected {len(emotions)} face(s)"
                )


                # =================================================
                # EMOTION TABLE
                # =================================================

                st.subheader(
                    "😊 Detected Emotions"
                )

                st.dataframe(
                    emotion_df,
                    use_container_width=True,
                    hide_index=True
                )


                # =================================================
                # MOST COMMON EMOTION
                # =================================================

                most_common_emotion = (
                    emotion_counts
                    .index[0]
                )


                st.metric(
                    "😊 Dominant Emotion",
                    most_common_emotion
                )


                # =================================================
                # EMOTION CHART
                # =================================================

                st.subheader(
                    "📊 Emotion Visualization"
                )

                st.bar_chart(
                    emotion_df.set_index(
                        "Emotion"
                    )["Count"]
                )


            else:

                st.warning(
                    "⚠️ No faces or emotions detected."
                )


        except Exception as e:

            st.error(
                "❌ Emotion detection failed."
            )

            st.write(
                "Please try another image with a clear face."
            )

            st.write(
                f"Error: {e}"
            )


# =========================================================
# IMAGE OPTIONS
# =========================================================

st.subheader(
    "📷 Choose an Image"
)

option = st.radio(
    "Select Image Source:",
    [
        "🧪 Try Sample Image",
        "📤 Upload My Own Image"
    ]
)


# =========================================================
# SAMPLE IMAGE
# =========================================================

if option == "🧪 Try Sample Image":

    if st.button(
        "🚀 Run AI Analysis",
        use_container_width=True
    ):

        with st.spinner(
            "📥 Downloading sample image..."
        ):

            response = requests.get(
                sample_image_url
            )

            image = Image.open(
                BytesIO(
                    response.content
                )
            ).convert("RGB")


        # =================================================
        # DISPLAY ORIGINAL IMAGE
        # =================================================

        st.subheader(
            "📷 Original Image"
        )

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )


        # =================================================
        # OBJECT DETECTION
        # =================================================

        results = detect_objects(
            image
        )


        detected_objects = (
            get_detected_objects(
                results
            )
        )


        show_object_results(
            results,
            detected_objects
        )


        # =================================================
        # EMOTION DETECTION
        # =================================================

        st.markdown("---")

        detect_emotions(
            image
        )


# =========================================================
# UPLOAD YOUR OWN IMAGE
# =========================================================

else:

    uploaded_image = st.file_uploader(
        "📤 Upload an Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")


        # =================================================
        # DISPLAY IMAGE
        # =================================================

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # =================================================
        # RUN AI ANALYSIS
        # =================================================

        if st.button(
            "🚀 Run AI Analysis",
            use_container_width=True
        ):


            # =================================================
            # OBJECT DETECTION
            # =================================================

            results = detect_objects(
                image
            )


            detected_objects = (
                get_detected_objects(
                    results
                )
            )


            show_object_results(
                results,
                detected_objects
            )


            # =================================================
            # EMOTION DETECTION
            # =================================================

            st.markdown("---")

            detect_emotions(
                image
            )
