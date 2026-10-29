# 📟 AI-Powered Customer Inquiry Tagger & Responder

A hybrid machine learning and GenAI triage system designed to optimize customer support operations.

## 👤 About Me & Project Vision
After **4 years of hands-on experience in Customer Service**, I transitioned into technology by completing my Machine Learning certificate. I built this project to directly bridge the gap between technical infrastructure and real-world customer problems. 

Traditional LLMs (like GPT-4) are amazing but expensive to run for every single ticket. This project demonstrates a production-minded balance: it uses a highly efficient, deterministic **locally trained classical ML model** to immediately tag intents, and applies **Generative AI** layers only for deeper sentiment analysis and dynamic contextual replies when credentials are provided.

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning Architecture:** Scikit-Learn (TF-IDF Vectorizer + Logistic Regression)
- **User Interface:** Streamlit Dashboard Framework
- **GenAI Tier:** OpenAI API (with structured JSON fallbacks)
