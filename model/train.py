import re
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ==========================
# Reproducibility
# ==========================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Load and Prepare Dataset
# ==========================
def load_data():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    fake["label"] = 1
    true["label"] = 0

    df = pd.concat([fake, true])
    df = df[["text", "label"]]
    df = df.dropna()
    df = df.drop_duplicates(subset="text")

    # Remove Reuters bias
    df["text"] = df["text"].str.replace(r"\(Reuters\).*?-", "", regex=True)
    df["text"] = df["text"].str.replace("Reuters", "", regex=True)

    return df.sample(15000, random_state=42).reset_index(drop=True)


# ==========================
# Custom Dataset Class
# ==========================
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ==========================
# Main Training Function
# ==========================
def main():
    df = load_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    val_encodings = tokenizer(
        val_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # Unfreeze last 2 layers
    for param in model.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./model/saved_model",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        save_strategy="epoch",
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(val_labels, preds)

    print("Final Validation Accuracy:", accuracy)

    model.save_pretrained("./model/saved_model")
    tokenizer.save_pretrained("./model/saved_model")


if __name__ == "__main__":
    main()
