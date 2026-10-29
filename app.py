import streamlit as st
import joblib
import os
from llm_helper import enrich_and_draft

st.set_page_config(page_title="AI Customer Support Agent", page_icon="🤖", layout="wide")

st.title("📟 AI-Powered Customer Inquiry Tagger & Responder")
st.markdown("---")

# 1. Load the localized core ML model binary file dynamically
@st.cache_resource
def load_ml_model():
    if os.path.exists('customer_model.pkl'):
        return joblib.load('customer_model.pkl')
    return None

model = load_ml_model()

# Handle pipeline status visualization
if model is None:
    st.error("⚠️ Core ML model file ('customer_model.pkl') not found! Please open a terminal and run `python ml_pipeline.py` first to train it.")
else:
    # FIX: Explicitly passed '2' into the columns function call
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Incoming Communication Input")
        sample_text = st.selectbox(
            "Quick Select a customer ticket sample:",
            [
                "Custom input...",
                "I am extremely upset. Where is my delivery? Order ID: #994A2",
                "Can you help me? I cannot login to my account on the desktop dashboard.",
                "I need an immediate refund for order 1290, the product arrived completely broken!"
            ]
        )
        
        if sample_text == "Custom input...":
            customer_input = st.text_area("Type customer message here:", placeholder="Type a message...")
        else:
            customer_input = st.text_area("Type customer message here:", value=sample_text)
            
        process_btn = st.button("Run Automation Triage Pipeline", type="primary")
        
    with col2:
        st.subheader("⚙️ Automated Triage Outputs")
        
        if process_btn and customer_input.strip():
            with st.spinner("Processing through hybrid ML + GenAI pipelines..."):
                # Step A: Deterministic ML Intent Prediction (Scikit-Learn)
                prediction = model.predict([customer_input])[0] # Grab the single string value
                
                st.metric(label="🎯 Predicted Intent (Traditional ML)", value=prediction)
                
                # Step B: GenAI Unstructured Context Enrichment & Resolution Generation
                ai_insights = enrich_and_draft(customer_input, prediction)
                
                # UI Layout presentation
                st.markdown("#### 💡 AI Operations Insights")
                # FIX: Explicitly passed '2' into the inner layout columns as well
                c_sub1, c_sub2 = st.columns(2)
                with c_sub1:
                    st.write(f"**Customer Sentiment:** {ai_insights.get('sentiment')}")
                with c_sub2:
                    st.write(f"**Extracted Reference ID:** {ai_insights.get('order_id')}")
                    
                st.markdown("#### 📝 Generated Automated Response Draft")
                st.info(ai_insights.get('reply_draft'))
        elif process_btn:
            st.warning("Please type or select a valid text sample first.")
