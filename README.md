# 🌱 PlantGuard AI

An AI-powered plant disease detection and information system designed to identify plant diseases and provide users with relevant disease information, symptoms, and remedies through a structured knowledge base.

## 📌 Overview

PlantGuard AI combines deep learning-based plant disease detection with a structured JSON-based information system. Instead of using a traditional chatbot model, the system retrieves disease-related information from predefined data to provide accessible and informative results.

## 🎯 Objectives

* Detect plant diseases using deep learning and computer vision.
* Provide information about identified diseases, symptoms, and remedies.
* Develop a simple and accessible plant health assistance system.
* Support early disease identification and informed plant care.
* Build a foundation for future AI-powered agricultural solutions.

## 🔍 What It Does

* **Plant Disease Detection:** Identifies plant diseases using trained deep learning models.
* **Disease Information Retrieval:** Fetches relevant information from a structured JSON knowledge base.
* **Symptom Identification:** Provides symptoms associated with detected diseases.
* **Remedy Suggestions:** Displays predefined remedies and management recommendations.
* **User Interaction:** Presents plant disease information in an easy-to-understand format.

> **Note:** The current system uses predefined JSON-based information retrieval rather than a generative AI chatbot.

## 💡 Motivation

Plant diseases can negatively affect crop health and agricultural productivity. Early identification and access to relevant disease information can help users take timely preventive and management measures.

PlantGuard AI was developed to explore how **computer vision and structured information retrieval** can be combined to create accessible plant health assistance tools.

## 🛠️ Tech Stack

| Technology                  | Purpose                                   |
## 🛠️ Tech Stack

- **Programming Language:** Python
- **Dataset Sources:** Kaggle, Google Images
- **Annotation & Preprocessing:** Roboflow
- **Object Detection:** YOLOv12
- **Data Storage:** JSON
- **Web Framework:** Streamlit


## 🔄 Workflow

```text
Input Plant Image
        │
        ▼
Image Preprocessing
        │
        ▼
Deep Learning Model
        │
        ▼
Plant Disease Detection
        │
        ▼
Disease Name Extraction
        │
        ▼
JSON Knowledge Base
        │
        ▼
Information Retrieval
        │
        ▼
Disease Information,
Symptoms & Remedies
```

### Workflow Explanation

1. **Image Input:** The user provides an image of a plant or affected leaf.
2. **Preprocessing:** The image is prepared for model inference.
3. **Disease Detection:** The trained deep learning model analyzes the image.
4. **Disease Identification:** The system extracts the predicted disease name.
5. **Information Retrieval:** The disease name is matched against the JSON knowledge base.
6. **Result Display:** The system provides relevant disease details, symptoms, and remedies.

## ⚠️ Limitations

* Detection performance depends on image quality, lighting, and dataset diversity.
* The model may struggle with diseases that are not represented in its training dataset.
* JSON-based information retrieval is limited to predefined disease information.
* The system does not currently use a generative AI model for conversational responses.
* **Detection accuracy may decrease in nighttime and dim-lighting conditions.**
* Remedy recommendations are predefined and should not replace professional agricultural consultation.

## 🚀 Future Work

* Integrate a **Retrieval-Augmented Generation (RAG)** pipeline for more flexible disease information retrieval.
* Develop a conversational interface using an LLM.
* Expand the dataset to include more crops, diseases, and real-world field conditions.
* Add multilingual support for regional agricultural users.
* Develop a mobile application for convenient field-level usage.
* Enable expert-verified treatment and disease management recommendations.


`plant-disease-detection` `deep-learning` `computer-vision` `python` `agriculture-ai` `yolo` `plant-health` `machine-learning` `json`
