import torch
import numpy as np
import shap
from transformers import BertTokenizer, BertForSequenceClassification


# ==========================
# Device Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Load Trained Model
# ==========================
model = BertForSequenceClassification.from_pretrained(
    "../model/saved_model"
)

tokenizer = BertTokenizer.from_pretrained(
    "../model/saved_model"
)

model.to(device)
model.eval()


# ==========================
# Prediction Wrapper
# ==========================
def predict_proba(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]

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


# ==========================
# SHAP Explainer
# ==========================
explainer = shap.Explainer(
    predict_proba,
    shap.maskers.Text(tokenizer)
)


# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    sample_text = "Breaking news: Government announces new economic reform policy."

    shap_values = explainer([sample_text])

    shap.plots.text(shap_values[0])
