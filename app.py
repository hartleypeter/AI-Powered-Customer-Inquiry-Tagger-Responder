import os
import pickle
import streamlit as st
from ml_pipeline import train_and_save_model
from llm_helper import analyze_and_draft_reply

st.set_page_config(page_title="AI Support Tagger", page_icon="🤖")
st.title("🤖 Smart Support Inbox")

model_path = "models/text_classifier.pkl"

# Automatically train the model if the file doesn't exist yet!
if not os.path.exists(model_path):
    with st.spinner("Initializing and training machine learning model..."):
        train_and_save_model()

with open(model_path, "rb") as f:
    ml_pipeline = pickle.load(f)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

user_input = st.text_area("Type customer message:", value="Hey my order #4491 hasn't arrived. I need this urgently!!")

if st.button("Process Message", type="primary"):
    predicted_category = ml_pipeline.predict([user_input])[0] # Get text prediction string
    ai_analysis = analyze_and_draft_reply(user_input, predicted_category)
    
    st.success("Processed!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Intent (ML)", predicted_category)
    col2.metric("Sentiment", ai_analysis['sentiment'])
    col3.metric("Order ID", ai_analysis['extracted_order_id'])
    
    st.markdown("### Draft Response:")
    st.info(ai_analysis['draft_reply'])
