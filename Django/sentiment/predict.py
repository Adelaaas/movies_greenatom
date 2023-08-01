import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tqdm import tqdm


# Предобработка данных и создание DataLoader
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, reviews, ratings, tokenizer, max_length):
        self.reviews = reviews
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = str(self.reviews[index])
        rating = float(self.ratings[index])

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating)
        }


# Класс создания датасет
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


# класс модели
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# функция тренировки модели
def train(model, data_loader, optimizer, scheduler, device):
    model.train()

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


# функция оценки модели
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# функция предсказания
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    # model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return "Положительный отзыв" if preds.item() == 1 else "Негативный отзыв"


def preprocess_review(review, tokenizer):
    # Предобработка отзыва и создание DataLoader
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True,
        return_token_type_ids=False
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask


def getPrediction(data):
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 128
    batch_size = 16
    num_epochs = 4
    learning_rate = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    model_path_bert_classifier = os.path.join(os.path.dirname(__file__), 'bert_classifier_imdb.pth')
    model_bert_clf = BERTClassifier(bert_model_name, num_classes).to(device)
    model_bert_clf.load_state_dict(torch.load(model_path_bert_classifier, map_location="cpu"))

    sentiment = predict_sentiment(data, model_bert_clf, tokenizer, device)

    model_path_bert_regressor = os.path.join(os.path.dirname(__file__), 'bert_regressor_imdb_v2.pth')
    model_bert_reg = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=1).to(device)
    model_bert_reg = model_bert_reg.float()
    model_bert_reg.load_state_dict(torch.load(model_path_bert_regressor, map_location="cpu"))
    
    input_ids, attention_mask = preprocess_review(data, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        output = model_bert_reg(input_ids, attention_mask=attention_mask)
        prediction = output.logits.item()
    # sentiment = 'Положительный отзыв'
    # prediction = 1.22222
    return sentiment, round(prediction, 2)