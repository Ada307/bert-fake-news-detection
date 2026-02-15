import torch
import numpy as np
import streamlit as st
import shap
from transformers import BertTokenizer, BertForSequenceClassification


# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 BERT Fake News Detection System")
st.write("Deep Learning + Explainable AI (SHAP)")


# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("adaj0610/bert-fake-news-detector")
    tokenizer = BertTokenizer.from_pretrained("adaj0610/bert-fake-news-detector")
    return model, tokenizer

model, tokenizer = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# ==========================
# Prediction Function
# ==========================
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item()
    prediction = torch.argmax(probs).item()

    return prediction, confidence


# ==========================
# SHAP Function
# ==========================
def explain(text):
    def predict_proba(texts):
        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.detach().cpu().numpy()

    explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
    shap_values = explainer([text])
    return shap_values


# ==========================
# UI
# ==========================
user_input = st.text_area("Enter News Text Below:", height=200, placeholder="Paste news article text here...")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred, conf = predict(user_input)

        if pred == 1:
            st.error(f"Prediction: FAKE ❌ (Confidence: {conf:.2f})")
        else:
            st.success(f"Prediction: REAL ✅ (Confidence: {conf:.2f})")

        st.subheader("Explainability (SHAP)")
        st.info("SHAP explanation available in local version.")
