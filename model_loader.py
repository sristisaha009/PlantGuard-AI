# model_loader.py
import sys
import os
import time
from typing import Tuple
from PIL import Image
import numpy as np
import streamlit as st
import torch

# ensure local yolov12 repo is importable (adjust path if your layout differs)
REPO_PATH = os.path.join(os.path.dirname(__file__), "yolov12")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)


class CompatiblePlantDiseaseModel:
    """
    Lazy-loading wrapper around the YOLO model used by the Streamlit app.
    Importing this module will NOT attempt to load the model or ultralytics.
    The heavy imports and model loading happen only when load_model() is called.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        # set default class names (optional)
        self.class_names = [
            "rice_brown_spot",
            "rice_bacterial_blight",
            "rose_black_spot",
            "tomato_early_blight",
            "healthy",
        ]

    def check_ultralytics_version(self):
        """Return ultralytics version string or None if not installed."""
        try:
            import ultralytics
            return getattr(ultralytics, "__version__", None)
        except Exception:
            return None

    def load_model(self, show_progress: bool = True):
        """
        Load the YOLO model. This function performs the actual import of ultralytics.
        Call this after importing the class (e.g. inside your app initialization).
        """
        if self.model is not None:
            return  # already loaded

        version = self.check_ultralytics_version()
        if version is None:
            raise RuntimeError("ultralytics package not found in this environment. "
                               "Make sure you installed the yolov12 fork or ultralytics used for training.")

        # Import YOLO here (lazy import)
        try:
            from ultralytics import YOLO
        except Exception as e:
            # re-raise with clearer message
            raise RuntimeError(f"Failed importing ultralytics.YOLO: {e}") from e

        # show progress in Streamlit if available
        try:
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Initializing YOLO model...")
                progress_bar.progress(30)
            else:
                progress_bar = None
                status_text = None

            # instantiate model
            self.model = YOLO(self.model_path)

            if show_progress:
                status_text.text("Testing model...")
                progress_bar.progress(70)

            # quick check that model has names
            if hasattr(self.model, "names") and self.model.names:
                # normalize names into our internal format optionally
                try:
                    # self.model.names may be dict {id: name}
                    names_list = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
                    st.success(f"✅ Model loaded! Classes: {names_list}")
                except Exception:
                    st.success("✅ Model loaded!")
            else:
                st.success("✅ Model loaded successfully!")

            if show_progress:
                progress_bar.progress(100)
                status_text.text("✅ Model ready!")
                time.sleep(0.8)
                progress_bar.empty()
                status_text.empty()

        except Exception as e:
            # make import-time errors visible and raise
            st.error(f"❌ Model loading failed: {e}")
            raise

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run inference and return (label_key, confidence).
        This function expects that load_model() has already been called successfully.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            image_np = np.array(image.convert("RGB"))

            results = self.model.predict(
                source=image_np,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

            # results[0] is expected to be an object with boxes, conf, cls
            if results and len(results) > 0:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy()

                    detections = []
                    for i in range(len(boxes)):
                        class_id = int(class_ids[i])
                        label = self.model.names[class_id] if hasattr(self.model, "names") else f"class_{class_id}"
                        detections.append({"label": label, "confidence": float(confidences[i])})

                    # return best detection
                    detections.sort(key=lambda x: x["confidence"], reverse=True)
                    best = detections[0]
                    return best["label"], best["confidence"]

            # default -> healthy
            return "healthy", 0.85

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            raise
