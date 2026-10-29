import os
import json
from openai import OpenAI

def enrich_and_draft(customer_text, predicted_intent):
    """
    Leverages GenAI structured prompts to perform sentiment analysis,
    extract relevant customer order identifiers, and draft responses.
    """
    # 1. Defensive programming block: Check if API key exists BEFORE initializing the client
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "sentiment": "Neutral (API Key Missing)",
            "order_id": "Not Extracted",
            "reply_draft": f"Hello! Thank you for your inquiry regarding '{predicted_intent}'. A support agent will review your text shortly."
        }

    try:
        # 2. Initialize the client safely now that we confirmed the key exists
        client = OpenAI()
        
        prompt = f"""
        You are an advanced AI Customer Support Operations Assistant. Analyze the customer inquiry text below.
        
        Inquiry: "{customer_text}"
        Predicted Intent Category: {predicted_intent}
        
        Provide a raw JSON response containing exactly these fields:
        1. "sentiment": (Strictly choose one: "Frustrated", "Neutral", "Satisfied")
        2. "order_id": (Extract any alphanumeric order code if present, otherwise output "None Found")
        3. "reply_draft": (Write a professional, highly empathetic first-touch automated message addressing their predicted intent)
        
        Respond with raw valid JSON only. Do not include markdown wraps or backticks.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        # Parse the JSON response payload safely
        result = json.loads(response.choices.message.content.strip())
        return result
    except Exception as e:
        return {
            "sentiment": "Error",
            "order_id": "Error",
            "reply_draft": f"Error running GenAI framework: {str(e)}"
        }
