# 📟 AI-Powered Customer Inquiry Tagger & Responder
### 🌐 Live Demo: [Test the Live App Here] https://ai-powered-customer-inquiry-tagger-responder-cksrwznldgqhit2mz.streamlit.app/

A hybrid machine learning and GenAI triage system designed to optimize customer support operations.

## 👤 About Me & Project Vision
After **4 years of hands-on experience in Customer Service**, I transitioned into technology by completing my Machine Learning certificate. I built this project to directly bridge the gap between technical infrastructure and real-world customer problems. 

Traditional LLMs (like GPT-4) are amazing but expensive to run for every single ticket. This project demonstrates a production-minded balance: it uses a highly efficient, deterministic **locally trained classical ML model** to immediately tag intents, and applies **Generative AI** layers only for deeper sentiment analysis and dynamic contextual replies when credentials are provided.

## 🛠️ Tech Stack
- **Language:** Python
- **Machine Learning Architecture:** Scikit-Learn (TF-IDF Vectorizer + Logistic Regression)
- **User Interface:** Streamlit Dashboard Framework
- **GenAI Tier:** OpenAI API (with structured JSON fallbacks)

## 🚀 How to Run This Project (Quick Start Guide)

If you are a beginner and want to see this application run live on your local computer, follow these simple steps using your terminal:

### 1. Clone the Repository
Download the project files to your machine:
```bash
git clone https://github.com/hartleypeter/AI-Powered-Customer-Inquiry-Tagger-Responder
cd AI-Powered-Customer-Inquiry-Tagger-Responder
```

### 2. Install the Required Software Packages
Make sure you have Python installed, then run this command to download the dependencies:
```bash
pip install pandas scikit-learn openai streamlit joblib
```

### 3. Train the Local Machine Learning Model
Before launching the application interface, you need to train the core classification engine:
```bash
python ml_pipeline.py
```
*(You will see a success message saying the model file has been saved locally!)*

### 4. Boot Up Your Interactive Web Dashboard
Run this final command to launch the visual dashboard interface in your web browser:
```bash
python -m streamlit run app.py
```

*💡 **Note on API Keys:** The application includes a defensive code architecture. It will run perfectly right out of the box using a local fallback framework even if you don't provide an OpenAI API key!*
