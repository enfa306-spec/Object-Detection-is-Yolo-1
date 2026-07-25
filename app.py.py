# =========================================================
# COMPUTER VISION - OBJECT DETECTION & COUNTING
# =========================================================

from ultralytics import YOLO
from PIL import Image
import pandas as pd
import streamlit as st
import requests
from io import BytesIO


st.markdown("---")

st.header("🖼️ Computer Vision - Object Detection & Counting")

st.write(
    "Use AI to detect and count objects inside an image."
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_yolo_model():

    model = YOLO("yolo11n.pt")

    return model


model = load_yolo_model()


# =========================================================
# SAMPLE IMAGE URL
# =========================================================

sample_image_url = (
    "https://ultralytics.com/images/bus.jpg"
)


# =========================================================
# IMAGE OPTIONS
# =========================================================

st.subheader("📷 Choose an Image")

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
        "🚀 Run Object Detection",
        use_container_width=True
    ):

        with st.spinner(
            "Downloading sample image..."
        ):

            response = requests.get(
                sample_image_url
            )

            image = Image.open(
                BytesIO(
                    response.content
                )
            )


        st.subheader(
            "📷 Sample Image"
        )

        st.image(
            image,
            caption="Sample Image",
            use_container_width=True
        )


        # =================================================
        # DETECT OBJECTS
        # =================================================

        with st.spinner(
            "🤖 AI is detecting objects..."
        ):

            results = model.predict(
                source=image,
                conf=0.25
            )


        # =================================================
        # DISPLAY DETECTION RESULT
        # =================================================

        result_image = results[
            0
        ].plot()


        st.subheader(
            "🎯 Detection Result"
        )

        st.image(
            result_image,
            caption="Detected Objects",
            use_container_width=True
        )


        # =================================================
        # GET DETECTED OBJECTS
        # =================================================

        detected_objects = []


        if results[0].boxes is not None:

            for cls in results[
                0
            ].boxes.cls:

                class_id = int(
                    cls
                )

                class_name = model.names[
                    class_id
                ]

                detected_objects.append(
                    class_name
                )


        # =================================================
        # COUNT OBJECTS
        # =================================================

        if len(
            detected_objects
        ) > 0:

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
            # DISPLAY TABLE
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
        )


        st.subheader(
            "📷 Uploaded Image"
        )


        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        if st.button(
            "🔍 Detect Objects",
            use_container_width=True
        ):

            with st.spinner(
                "🤖 AI is detecting objects..."
            ):

                results = model.predict(
                    source=image,
                    conf=0.25
                )


            # Detection Result

            result_image = results[
                0
            ].plot()


            st.subheader(
                "🎯 Detection Result"
            )


            st.image(
                result_image,
                caption="Detected Objects",
                use_container_width=True
            )


            # Get Objects

            detected_objects = []


            if results[0].boxes is not None:

                for cls in results[
                    0
                ].boxes.cls:

                    class_id = int(
                        cls
                    )

                    class_name = model.names[
                        class_id
                    ]

                    detected_objects.append(
                        class_name
                    )


            # Count Objects

            if len(
                detected_objects
            ) > 0:

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


                st.subheader(
                    "📊 Detected Objects Count"
                )


                st.dataframe(
                    object_df,
                    use_container_width=True,
                    hide_index=True
                )


                total_objects = len(
                    detected_objects
                )


                st.metric(
                    "🔢 Total Objects Detected",
                    total_objects
                )


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