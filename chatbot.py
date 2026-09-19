import json
import os
import re
import streamlit as st

class DiseaseChatbot:
    def __init__(self, knowledge_path="disease_knowledge.json"):
        """Initialize chatbot with a reliable JSON path."""
        # Always resolve the full path relative to this file
        self.knowledge_path = os.path.join(os.path.dirname(__file__), knowledge_path)
        self.knowledge = self.load_knowledge()

    def load_knowledge(self):
        """Load disease knowledge base from JSON file."""
        if not os.path.exists(self.knowledge_path):
            st.error(f"❌ Knowledge file not found at {self.knowledge_path}")
            return {}
        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalize keys in JSON for consistent lookup
            normalized_data = {
                self.normalize_key(k): v for k, v in data.items()
            }
            return normalized_data
        except Exception as e:
            st.error(f"⚠️ Error loading knowledge base: {e}")
            return {}

    def normalize_key(self, disease_name):
        """Convert disease name or label to standardized key format."""
        if not disease_name:
            return None
        key = disease_name.lower()
        # Remove special characters, hyphens, spaces, etc.
        key = re.sub(r"[^a-z0-9]+", "_", key)
        key = key.strip("_")
        return key

    def answer(self, detected_disease, user_input, chat_history):
        """Generate chatbot response based on the detected disease and query intent."""
        default_reply = (
            "I couldn't find information about this specific disease in my knowledge base. "
            "If you uploaded an image, please make sure it's clear. You can ask about symptoms, "
            "treatment, prevention, or causes for plant diseases."
        )

        # Normalize YOLO label to match JSON keys
        key = self.normalize_key(detected_disease)

        if not key or key not in self.knowledge:
            st.warning(f"⚠️ No match found for '{detected_disease}' → normalized as '{key}'")
            return {"reply": default_reply}

        disease_info = self.knowledge[key]
        user_input_lower = user_input.lower()

        # --- Intent Recognition ---
        if "cause" in user_input_lower:
            reply = f"🦠 **Cause:** {disease_info.get('cause', 'Information not available.')}"
        elif "symptom" in user_input_lower:
            symptoms = disease_info.get("symptoms", [])
            reply = "📋 **Symptoms:**\n- " + "\n- ".join(symptoms) if symptoms else "No symptom data available."
        elif "treat" in user_input_lower or "cure" in user_input_lower:
            treatment = disease_info.get("treatment", [])
            reply = "💊 **Treatment Methods:**\n- " + "\n- ".join(treatment) if treatment else "No treatment info found."
        elif "prevent" in user_input_lower or "avoid" in user_input_lower:
            prevention = disease_info.get("prevention", [])
            reply = "🛡️ **Prevention Tips:**\n- " + "\n- ".join(prevention) if prevention else "No prevention info found."
        else:
            reply = f"🌿 **{disease_info.get('name', 'Unknown Disease')}**\n\n{disease_info.get('additional_info', 'No additional info available.')}"

        return {"reply": reply}
