## BERT-Based Fake News Detection System
A Deep Learning-based Fake News Detection system built using BERT with SHAP explainability and deployed using Streamlit.

## Features
- Fine-tuned BERT (classifier head only)
- Confidence score prediction
- SHAP-based word-level explainability
- Streamlit web interface
- Modular project structure

**Dataset**
ISOT Fake News Dataset with 44000 labeled news articles (Fake/Real)
For faster experimentation, a 5000-sample subset was used for training.

## Model Architecture
BERT (Frozen Encoder)  
 Dropout  
 Linear Classification Layer  
 Softmax  
Only the classification head was trained to reduce training time.

**Tech Stack**
- Python
- PyTorch
- HuggingFace Transformers
- SHAP
- Streamlit
- Google Colab (GPU)
- Full fine-tuning of BERT

## Performance
- Validation Accuracy: ~90% (depending on subset)
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Attention visualization
- Cloud deployment (AWS / HuggingFace Spaces)
