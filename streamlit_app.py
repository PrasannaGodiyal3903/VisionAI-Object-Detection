import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="VisionAI",
    page_icon="🤖",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

h1,h2,h3,h4 {
    color:white;
}

.stButton>button {
    width:100%;
    height:55px;
    border-radius:15px;
    background:#2563eb;
    color:white;
    font-size:18px;
    border:none;
}

.stButton>button:hover {
    background:#1d4ed8;
}

.block-container {
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------

st.title("🤖 VisionAI")
st.caption("Real-Time Object Detection using YOLOv8 + FastAPI")

# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.title("⚙ VisionAI")

    st.markdown("---")

    st.write("Built using")

    st.success("YOLOv8")
    st.success("FastAPI")
    st.success("Streamlit")
    st.success("Python 3.13")

    st.markdown("---")

    st.info(
        "Upload an image and detect objects in real time."
    )

# -------------------- INPUT --------------------

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)
st.caption(f"Current Confidence Threshold: **{confidence:.2f}**")

# -------------------- MAIN --------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Detect Objects"):

        files = {
            "file": uploaded_file.getvalue()
        }

        with st.spinner("Running YOLOv8..."):

            img_response = requests.post(
                "http://127.0.0.1:8001/img_object_detection_to_img",
                params={
                    "confidence": confidence
                },
                files=files
            )

            json_response = requests.post(
                "http://127.0.0.1:8001/img_object_detection_to_json",
                params={
                    "confidence": confidence
                },
                files=files
            )

        if img_response.status_code == 200 and json_response.status_code == 200:

            detected = Image.open(io.BytesIO(img_response.content))

            result = json_response.json()

            detections = pd.DataFrame(result["detect_objects"])

            with col2:
                st.subheader("🎯 Detection Result")
                st.image(detected, use_container_width=True)

            st.success("Detection Completed Successfully")

            st.download_button(
                "⬇ Download Result",
                img_response.content,
                file_name="detected.jpg",
                mime="image/jpeg"
            )

            st.divider()

            st.subheader("📊 Detection Statistics")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Objects Found",
                len(detections)
            )

            c2.metric(
                "Average Confidence",
                f"{detections['confidence'].mean()*100:.1f}%"
            )

            c3.metric(
                "Highest Confidence",
                f"{detections['confidence'].max()*100:.1f}%"
            )

            st.divider()

            st.subheader("🎯 Objects Detected")

            for _, row in detections.iterrows():

                conf = row["confidence"] * 100

                if conf >= 90:
                    color = "#22C55E"
                elif conf >= 70:
                    color = "#FACC15"
                else:
                    color = "#EF4444"

                st.markdown(
                    f"""
<div style="
background:#1e293b;
padding:18px;
border-radius:15px;
margin-bottom:15px;
border-left:8px solid {color};
">

<h3 style="color:white;margin:0;">
📦 {row['name'].title()}
</h3>

<p style="font-size:18px;color:#CBD5E1;margin-top:8px;">
Confidence:
<b style="color:{color};">
{conf:.2f}%
</b>
</p>

</div>
""",
                    unsafe_allow_html=True
                )

        else:

            st.error("Unable to connect to FastAPI server.")

# -------------------- FOOTER --------------------

st.markdown("---")

st.caption("Built with ❤️ using Streamlit, FastAPI and YOLOv8")