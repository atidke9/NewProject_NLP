# ============Imports===============
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, ElectraForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# =================Hyperparameters=================
num_epochs = 3
checkpoint1 = 'distilbert-base-uncased'
checkpoint2 = 'google/electra-small-discriminator'
max_len = 512
model1 = 'distilbert'
model2 = 'electra'
model3 = 'ensemble'
model_name = model1

# ===========Load data and preprocess ==============

train_df = pd.read_csv('train.csv')[:140000]
train_df.columns = ['label','text']
train_df['label'] = train_df['label'] - 1
train_texts = train_df['text'].values.tolist()
train_labels = train_df['label'].values.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# =================Tokenizing and creating data loaders=====================
tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint1)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = YelpDataset(train_encodings, train_labels)
val_dataset = YelpDataset(val_encodings, val_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=6)
eval_dataloader = DataLoader(val_dataset, batch_size=6)

# ==========================Models============================================

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.m1 = DistilBertForSequenceClassification.from_pretrained(checkpoint1)
        self.m2 = ElectraForSequenceClassification.from_pretrained(checkpoint2)
        self.dropout = nn.Dropout(0.3)
        self.out3 = nn.Linear(4,2)
    def forward(self, ids):
        output_1 = self.m1(ids, return_dict=False)
        output_2 = self.dropout(output_1[0])
        output_3 = self.m2(ids, return_dict=False)
        output_4 = self.dropout(output_3[0])
        output_5 = torch.cat((output_2, output_4), dim=1)
        output = self.out3(output_5)
        return output

if model_name == model1:
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint1)
elif model_name == model2:
    model = ElectraForSequenceClassification.from_pretrained(checkpoint2)
elif model_name == model3:
    model = Classifier()
print(model, file=open('summary_{}.txt'.format(model_name), "w"))
model.to(device)
optim = AdamW(model.parameters(), lr=5e-5)
criterian = nn.CrossEntropyLoss()
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)
progress_bar = tqdm(range(num_training_steps))

# =======================Model Training=====================
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        if (model_name == model1) | (model_name == model2):
            outputs = model(input_ids, labels = labels)
            loss = outputs[0]
        else:
            outputs = model(input_ids)
            loss = criterian(outputs, labels)
        loss.backward()
        optim.step()
        lr_scheduler.step()
        progress_bar.update(1)


PRED = []
Y = []
def update(pred,y):
    x = pred.detach().cpu().numpy()
    z = y.detach().cpu().numpy()
    for i in range(len(x)):
        PRED.append(x[i])
        Y.append(z[i])

num_eval_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()
for batch in eval_dataloader:
    torch.no_grad()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    if (model_name == model1) | (model_name == model2):
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
    else:
        outputs = model(input_ids)
        logits = outputs
    predictions = torch.argmax(logits, dim=-1)
    update(predictions,labels)
    progress_bar.update(1)

acc = accuracy_score(Y, PRED)
print("\nValidation accuracy:",acc)

pre = precision_score(Y, PRED)
print("\nValidation precision:",pre)

rec = recall_score(Y, PRED)
print("\nValidation recall:",rec)

if os.path.exists("model_{}.pt".format(model_name)):
    os.remove("model_{}.pt".format(model_name))
print('Old model removed!')
torch.save(model.state_dict(), "model_{}.pt".format(model_name))
print('New model saved')