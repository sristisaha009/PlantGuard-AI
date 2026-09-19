import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import sys
sys.path.append("yolov12")  # path to your cloned YOLOv12 repo
from ultralytics import YOLO


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- MODERN CSS ----------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2E8B57, #32CD32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .tagline {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        height: 100%;
        transition: transform 0.3s ease;
    }
    .card:hover { transform: translateY(-5px); }
    .card-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2E8B57;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .stButton button {
        background: linear-gradient(135deg, #2E8B57, #228B22);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.4);
    }
    .chat-container {
        background: linear-gradient(135deg, #f8fff8, #f0f8ff);
        border-radius: 20px;
        padding: 1.5rem;
        height: 500px;
        overflow-y: auto;
        border: 2px solid #e8f5e8;
    }
    .user-message {
        background: linear-gradient(135deg, #2E8B57, #228B22);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 5px 25px;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    }
    .bot-message {
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 25px 5px;
        margin: 0.8rem 0;
        max-width: 80%;
        border: 2px solid #e8f5e8;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .results-card {
        background: linear-gradient(135deg, #f0fff0, #e8f5e8);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid #2E8B57;
        text-align: center;
    }
    .disease-name { font-size: 2rem; font-weight: 800; color: #2E8B57; margin: 1rem 0; }
    .confidence-high { color: #28a745; font-weight: 800; font-size: 1.3rem; }
    .confidence-medium { color: #ffc107; font-weight: 800; font-size: 1.3rem; }
    .confidence-low { color: #dc3545; font-weight: 800; font-size: 1.3rem; }
    .quick-actions { display: flex; gap: 10px; margin-top: 1.5rem; flex-wrap: wrap; }
    .status-success {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .status-error {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- STATE SETUP ----------------------
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_disease' not in st.session_state:
        st.session_state.current_disease = None
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    if 'detection_confidence' not in st.session_state:
        st.session_state.detection_confidence = 0.0
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""


# ---------------------- MAIN APP ----------------------
def main():
    initialize_session_state()

    st.markdown('<h1 class="main-header">🌿 PlantGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">AI-powered plant disease detection and expert guidance</p>', unsafe_allow_html=True)

    try:
        from model_loader import CompatiblePlantDiseaseModel
        from chatbot import DiseaseChatbot

        chatbot = DiseaseChatbot()
        model_loaded = False
        model_path = r"best.pt"

        if not os.path.exists(model_path):
            st.markdown('<div class="status-error">❌ Model file not found. Please check the file path.</div>', unsafe_allow_html=True)
            st.stop()

        with st.spinner("🔄 Initializing AI detection engine..."):
            try:
                detector = CompatiblePlantDiseaseModel(model_path)
                detector.load_model()  # ✅ <-- THIS loads the YOLOv12 model
                model_loaded = True
            except Exception as e:
                st.markdown(f'<div class="status-error">❌ AI Engine Initialization Failed: {str(e)}</div>',
                            unsafe_allow_html=True)
                model_loaded = False

        if not model_loaded:
            st.markdown("""
            <div class="status-error">
            <h4>🚨 System Initialization Required</h4>
            <p>Please ensure:</p>
            <ul>
                <li>AI model file is available</li>
                <li>Required dependencies are installed</li>
                <li>System has sufficient resources</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    except Exception as e:
        st.markdown(f'<div class="status-error">❌ System Error: {str(e)}</div>', unsafe_allow_html=True)
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    # ------------------ LEFT COLUMN ------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🔍 Plant Health Scanner</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h4 style="color: #2E8B57; margin-bottom: 1rem;">📤 Upload Plant Image</h4>
            <p style="color: #666;">Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Captured Image", width='stretch')
            st.session_state.uploaded_image = image

            if st.button("🚀 Analyze Plant Health", use_container_width=True):
                with st.spinner("🔬 Scanning for diseases..."):
                    try:
                        disease, confidence = detector.predict(image)
                        st.session_state.current_disease = disease
                        st.session_state.detection_confidence = confidence
                        st.session_state.detection_done = True

                        st.markdown('<div class="status-success">✅ Analysis Complete!</div>', unsafe_allow_html=True)

                        if confidence > 0.8:
                            conf_class = "confidence-high"; status = "🟢 High Confidence"
                        elif confidence > 0.6:
                            conf_class = "confidence-medium"; status = "🟡 Medium Confidence"
                        else:
                            conf_class = "confidence-low"; status = "🔴 Low Confidence"

                        st.markdown(f"""
                        <div class="results-card">
                            <h3>📊 Detection Results</h3>
                            <div class="disease-name">{disease.replace('_', ' ').title()}</div>
                            <p><strong>Confidence Level:</strong> <span class="{conf_class}">{confidence:.2%}</span></p>
                            <p><strong>Status:</strong> {status}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(f'<div class="status-error">❌ Analysis Failed: {str(e)}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ RIGHT COLUMN ------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">💬 Plant Health Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; color: #666; padding: 2rem;">
                <h4>👋 Welcome to PlantGuard AI!</h4>
                <p>I'm your plant health assistant. Upload an image to get started, or ask me about:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Plant disease symptoms</li>
                    <li>Treatment options</li>
                    <li>Prevention methods</li>
                    <li>General plant care</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                css_class = "user-message" if message["role"] == "user" else "bot-message"
                st.markdown(f'<div class="{css_class}">{message["content"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ------------------ Chat Input ------------------
        user_input = st.text_input(
            "Type your question...",
            placeholder="Ask about plant diseases, treatments, or prevention...",
            key="chat_input",
            label_visibility="collapsed"
        )

        # ✅ Initialize control flags
        if "input_ready" not in st.session_state:
            st.session_state.input_ready = False
        if "last_message" not in st.session_state:
            st.session_state.last_message = None

        # ✅ Step 1: Detect when the user actually enters something new
        if user_input and not st.session_state.input_ready and user_input != st.session_state.last_message:
            st.session_state.pending_input = user_input
            st.session_state.input_ready = True
            st.rerun()

        # ✅ Step 2: Process pending input safely (only once)
        elif st.session_state.input_ready and "pending_input" in st.session_state:
            msg = st.session_state.pop("pending_input")
            st.session_state.input_ready = False
            st.session_state.last_message = msg  # Remember last message

            # Process user message
            st.session_state.chat_history.append({"role": "user", "content": msg})
            response = chatbot.answer(st.session_state.current_disease, msg, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "bot", "content": response["reply"]})

            st.rerun()

        # ------------------ Quick Actions ------------------
        if st.session_state.detection_done and st.session_state.current_disease != "healthy":
            st.markdown("### 💡 Quick Actions")
            st.markdown('<div class="quick-actions">', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🦠 Causes", use_container_width=True):
                    st.session_state["pending_input"] = "What causes this disease?"
                    st.session_state.input_ready = True
                    st.rerun()

            with col2:
                if st.button("📋 Symptoms", use_container_width=True):
                    st.session_state["pending_input"] = "What are the symptoms?"
                    st.session_state.input_ready = True
                    st.rerun()

            with col3:
                if st.button("💊 Treatment", use_container_width=True):
                    st.session_state["pending_input"] = "How to treat it?"
                    st.session_state.input_ready = True
                    st.rerun()

            with col4:
                if st.button("🛡️ Prevention", use_container_width=True):
                    st.session_state["pending_input"] = "How to prevent it?"
                    st.session_state.input_ready = True
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Footer ------------------
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #666; border-top: 1px solid #e0e0e0;">
        <p>🌱 PlantGuard AI - Protecting your plants with artificial intelligence</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
