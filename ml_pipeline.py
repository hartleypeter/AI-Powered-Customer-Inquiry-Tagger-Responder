import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_model():
    print("🤖 Preparing customer service training dataset...")
    
    # 1. Synthesize explicit customer support domain text data
    data = {
        'text': [
            "Can I get a refund for my late order?", "I want my money back immediately", 
            "Where is my package? It hasn't arrived", "Track my delivery status please",
            "The app crashes when I try to login", "I cannot reset my password on the screen",
            "My item arrived broken, I need a replacement", "Is my order shipped yet?",
            "Having trouble connecting my account to the server", "Return request for order 4920"
        ],
        'intent': [
            'Refund Request', 'Refund Request', 
            'Delivery Status', 'Delivery Status',
            'Technical Support', 'Technical Support',
            'Refund Request', 'Delivery Status',
            'Technical Support', 'Refund Request'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 2. Build the standard Scikit-Learn pipeline
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=200))
    ])
    
    # 3. Train the core architecture
    print("🏋️ Training traditional machine learning model...")
    model_pipeline.fit(df['text'], df['intent'])
    
    # 4. Save the binary model parameters locally
    joblib.dump(model_pipeline, 'customer_model.pkl')
    print("💾 Model saved successfully as 'customer_model.pkl'!")

if __name__ == "__main__":
    train_model()
