
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🚧"
)

st.title("🚧 Pothole Detection and Alert System")
st.write("Upload a road image to detect potholes.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    0.10, 0.90, 0.35
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image",
             use_container_width=True)

    if st.button("Detect Potholes"):

        result = model.predict(
            image,
            conf=confidence,
            verbose=False
        )[0]

        count = len(result.boxes)

        output = result.plot()[:, :, ::-1]

        st.subheader("Detection Result")
        st.image(output, use_container_width=True)

        st.metric("Potholes Detected", count)

        if count > 0:
            st.error("⚠️ Pothole Detected! Drive Carefully.")
        else:
            st.success("No pothole detected.")

st.caption("Built using Python, YOLOv8 and Streamlit")
