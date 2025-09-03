import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

true_path = r"C:\Users\user\Documents\Downloads\NewsDataExtracted\True.csv"
fake_path = r"C:\Users\user\Documents\Downloads\NewsDataExtracted\Fake.csv"

df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

df_true = df_true.sample(n = 10000, random_state = 76)
df_fake = df_fake.sample(n = 10000, random_state = 76)

df_true['label'] = 1
df_fake['label'] = 0

data = pd.concat([df_true, df_fake])
data = data.sample(frac = 1).reset_index(drop = True)

data = data.drop(columns = ['subject', 'date'])
data['content'] = data['title'] + " " + data['text']
data = data.drop(columns = ['title', 'text'])

data = data.drop_duplicates(subset = "content").reset_index(drop = True)

# Now split
x_train, x_test, y_train, y_test = train_test_split(
    data["content"], data["label"],
    test_size=0.2, random_state=62, stratify=data["label"]
)



import torch
from torch.utils.data import dataset
from torch.utils.data import Dataset

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # or however many classes you have
)

model.to(device)
print("Model Loaded Successfully")


train_encodings = tokenizer(list(x_train), truncation = True , padding = True , max_length = 512)
test_encodings = tokenizer(list(x_test), truncation = True , padding = True , max_length = 512)
print("Encodings Done")

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
    def __len__(self):
        return len(self.labels)



train_dataset = NewsDataset(train_encodings, y_train)
test_dataset = NewsDataset(test_encodings , y_test)
print("DataSet Done")

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size = 6, shuffle = True )
test_loader = DataLoader(test_dataset, batch_size = 6, shuffle = True)

print(len(train_loader))
print(len(test_loader))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import BertForSequenceClassification

from transformers import BertForSequenceClassification



# Freeze embedding layer
for param in model.distilbert.parameters():
    param.requires_grad = False

for layer in model.distilbert.transformer.layer[:4]:  # freeze first 4 layers out of 6
    for param in layer.parameters():
        param.requires_grad = False

print("Model layers frozen")

from transformers import get_linear_schedule_with_warmup

# Optimizer + Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_training_steps = len(train_loader) * 2  # epochs = 3
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)

# Validation Function
def validate(model, device, test_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(test_loader), acc, f1


# Training Loop
def train(model, train_loader, optimizer, scheduler, device, epochs):
    print("Training Started")

    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("Loss :", loss.item())
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Training metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="macro")

        # Validation metrics
        val_loss, val_acc, val_f1 = validate(model, device, test_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")


# Run training
train(model, train_loader, optimizer, scheduler, device, epochs=2)

# Save model
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")


