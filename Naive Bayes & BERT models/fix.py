import json

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from a JSONL file
def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Load data from the provided JSONL files
train_data = load_data_from_jsonl('data/train.jsonl')
test_data = load_data_from_jsonl('data/test.jsonl')

# Extract texts and labels for training and testing
train_texts = [entry['text'] for entry in train_data]
train_labels = [entry['label'] for entry in train_data]
test_texts = [entry['text'] for entry in test_data]
test_labels = [entry['label'] for entry in test_data]

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class for handling the text data
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))

# Function to train and evaluate the model
def train_eval_bert_model(model, train_dataset, test_dataset, epochs=2, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        # Wrap the train_loader with tqdm for a progress bar
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    actuals = []

    # Wrap the test_loader with tqdm for a progress bar
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            actuals.extend(labels.tolist())

    # Calculate accuracy
    accuracy = accuracy_score(actuals, predictions)
    return accuracy, predictions


# Train and evaluate the model
accuracy, predictions = train_eval_bert_model(model, train_dataset, test_dataset)
print("Model Accuracy:", accuracy)

correctList = []
incorrectList = []

# Ensure test_labels is a list or convert it to a list
test_labels_list = list(test_labels)

for i in range(len(test_labels_list)):
    if test_labels_list[i] == predictions[i]:
        if len(correctList) < 20:
            correctList.append(test_texts[i])
    else:
        if len(incorrectList) < 12:
            incorrectList.append(test_texts[i])

    if len(correctList) >= 20 and len(incorrectList) >= 12:
        break

print("Correctly labeled list: ")
print(*correctList, sep = "\n")
print("\n")
print("Incorrectly labeled list: ")
print(*incorrectList, sep = "\n")

import string

def split_sentence(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    ans = " ".join(words[3:16])
    return ans

def get_layer(layer, attention, shape):
  attention = attention[layer][0][layer].detach()/shape
  return attention

#chosen layer to generate attention matrix
layer = 6
def createAttentionMatrix(review, layer, color, index, correct=True):
    review = split_sentence(review)
    print(review)
    tk = [['[CLS]'] + tokenizer.tokenize(t)[:768] for t in [review]]
    tks = torch.tensor([tokenizer.convert_tokens_to_ids(tokens) for tokens in tk], dtype=torch.int)
    tks = torch.nn.utils.rnn.pad_sequence(tks, batch_first=True, padding_value=0)
    
    # Enable the output of attention weights
    model_output = model.cpu()(tks, output_attentions=True)
    attentionOutput = model_output.attentions

    att = get_layer(layer, attentionOutput, tks.shape[1])
    plt.figure(figsize=(16,5))
    heat = sns.heatmap(att, annot=True, cmap=sns.light_palette(color, as_cmap=True), linewidths=1, xticklabels=tk[0], yticklabels=tk[0])
    if correct:
        heat.set_title('Attention matrix for correctly labeled index ' + index)
    else:
        heat.set_title('Attention matrix for incorrectly labeled index ' + index)


# Visualization of Attention Matrices
for i, text in enumerate(correctList[:8]):
    createAttentionMatrix(text, layer, 'green', str(i), correct=True)


for i, text in enumerate(incorrectList[:8]):
    createAttentionMatrix(text, layer, 'red', str(i), correct=False)