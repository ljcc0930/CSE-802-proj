import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

from utils import AverageMeter


def svm(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    model.final_predict = lambda X: model.predict(scaler.transform(X))
    return model


def naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    model.final_predict = lambda X: model.predict(X)
    return model


def train_nn(model, X_train, y_train, batch_size=64, epochs=10):
    device = next(model.parameters()).device
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trange = tqdm.trange(epochs, desc="Epoch")

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for epoch in trange:
        avg_loss.reset()
        avg_acc.reset()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item(), inputs.size(0))
            avg_acc.update(((outputs > 0.5) == targets).sum().item(), inputs.size(0))
        trange.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)


def test_nn(model, X, batch_size=64):
    device = next(model.parameters()).device
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            logits = outputs.detach().cpu().numpy()

            predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)
    predicted_labels = predictions > 0.5
    return predicted_labels


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)


def lstm(X_train, y_train, max_features=100, batch_size=64, epochs=10):
    model = LSTMModel(max_features, 256, 1)
    model = model.cuda()
    train_nn(model, X_train, y_train, batch_size, epochs)
    model.final_predict = lambda X: test_nn(model, X, batch_size)
    return model


class TextCNN(nn.Module):
    def __init__(self, embed_dim, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n, (f, embed_dim)) for f, n in zip(filter_sizes, num_filters)]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(sum(num_filters), sum(num_filters))
        self.fc2 = nn.Linear(sum(num_filters), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)


def text_cnn(X_train, y_train, max_features=100, batch_size=64, epochs=10):
    model = TextCNN(max_features, 1, [3, 4, 5], [256, 256, 256])
    model = model.cuda()
    train_nn(model, X_train, y_train, batch_size, epochs)
    model.final_predict = lambda X: test_nn(model, X, batch_size)
    return model


def encode_bert(tokenizer, X):
    input_ids = []
    attention_masks = []
    for x in X:
        encoded_dict = tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=200,
            pad_to_max_length=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    return torch.tensor(input_ids), torch.tensor(attention_masks)


def train_bert(model, X, y, batch_size=64, epochs=10):
    device = next(model.parameters()).device
    input_ids, attention_masks = encode_bert(model.tokenizer, X)
    dataset = TensorDataset(input_ids, attention_masks, torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    trange = tqdm.trange(epochs, desc="Epoch")

    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    for epoch in trange:
        avg_loss.reset()
        avg_acc.reset()
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item(), b_input_ids.size(0))
            avg_acc.update(
                (outputs.logits.argmax(axis=1) == b_labels).sum().item(),
                b_input_ids.size(0),
            )
            trange.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, item=avg_acc.count)


def test_bert(model, X, batch_size=64):
    device = next(model.parameters()).device
    input_ids, attention_masks = encode_bert(model.tokenizer, X)
    dataset = TensorDataset(input_ids, attention_masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            b_input_ids, b_input_mask = batch

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()

            predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)
    predicted_labels = np.argmax(predictions, axis=1)

    return predicted_labels


def bert(X_train, y_train, batch_size=64, epochs=10):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )
    model.tokenizer = tokenizer
    model = model.cuda()
    train_bert(model, X_train, y_train, batch_size, epochs)
    model.final_predict = lambda X: test_bert(model, X, batch_size)
    return model


def roberta(X_train, y_train, batch_size=64, epochs=10):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base"
    )
    model.tokenizer = tokenizer
    model = model.cuda()
    train_bert(model, X_train, y_train, batch_size, epochs)
    model.final_predict = lambda X: test_bert(model, X, batch_size)
    return model


def opt(X_train, y_train, batch_size=8, epochs=10):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-1.3b")
    model.tokenizer = tokenizer
    model = model.cuda()
    train_bert(model, X_train, y_train, batch_size, epochs)
    model.final_predict = lambda X: test_bert(model, X, batch_size)
    return model


def train_classifier(method, X_train, y_train, max_features=1000, batch_size=64, epochs=10):
    if method == "SVM":
        return svm(X_train, y_train)
    elif method == "NaiveBayes":
        return naive_bayes(X_train, y_train)
    elif method == "LSTM":
        return lstm(X_train, y_train, max_features, batch_size, epochs)
    elif method == "TextCNN":
        return text_cnn(X_train, y_train, max_features, batch_size, epochs)
    elif method == "BERT":
        return bert(X_train, y_train, batch_size, epochs)
    elif method == "RoBERTa":
        return roberta(X_train, y_train, batch_size, epochs)
    elif method == "OPT":
        return opt(X_train, y_train, batch_size, epochs)
    else:
        raise ValueError("Unsupported classifier method.")


def test_classifier(model, X_test, y_test):
    predictions = model.final_predict(X_test)
    return classification_report(y_test, predictions), predictions
