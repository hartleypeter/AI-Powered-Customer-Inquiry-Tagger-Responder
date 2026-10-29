import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Self-contained Demo Response Function (Removes llm_helper dependency)
def analyze_and_draft_reply_local(customer_message, detected_category):
    # Standard Fallback Demo Mode for free usage
    return {
        "sentiment": "Neutral (Demo Mode)",
        "extracted_order_id": "N/A",
        "draft_reply": f"Hi! Thank you for messaging us regarding your inquiry about '{detected_category}'. A representative will assist you shortly."
    }

st.set_page_config(page_title="AI Support Tagger", page_icon="🤖")
st.title("🤖 Smart Support Inbox")

model_path = "models/text_classifier.pkl"

# 2. All-in-one training logic directly inside app.py
if not os.path.exists(model_path):
    with st.spinner("Initializing and training machine learning model..."):
        data = {
            "text": [
                "Hey, where is my order? It has been a week.",
                "Can I get a refund for my item? It arrived broken.",
                "Hi, do you sell this dress in blue or size M?",
                "My tracking number says invalid, can you check?",
                "I want my money back, this product doesn't work.",
                "Are you guys open on Sundays? What are your hours?",
                "Is the promo code still active for the shoes?",
                "I need to cancel my order before it ships out.",
                "Where is your shop located? Can I walk in?",
                "The app keeps crashing when I checkout. Please help."
            ],
            "category": [
                "Delivery Status", "Refund Request", "Product Inquiry", "Delivery Status", "Refund Request",
                "General Inquiry", "Product Inquiry", "Order Modification", "General Inquiry", "Technical Support"
            ]
        }
        df = pd.DataFrame(data)
        X, y = df["text"], df["category"]
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
            ('classifier', LogisticRegression(C=1.0, max_iter=1000))
        ])
        pipeline.fit(X, y)
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

# Load the verified pipeline weights
with open(model_path, "rb") as f:
    ml_pipeline = pickle.load(f)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

user_input = st.text_area("Type customer message:", value="Hey my order #4491 hasn't arrived. I need this urgently!!")

if st.button("Process Message", type="primary"):
    predicted_category = ml_pipeline.predict([user_input])[0] # Get single string prediction
    ai_analysis = analyze_and_draft_reply_local(user_input, predicted_category)
    
    st.success("Processed!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Intent (ML)", str(predicted_category))
    col2.metric("Sentiment", ai_analysis['sentiment'])
    col3.metric("Order ID", ai_analysis['extracted_order_id'])
    
    st.markdown("### Draft Response:")
    st.info(ai_analysis['draft_reply'])
