import os
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



# PATH = os.getcwd() + '/DeepLearning/Deep-Learning/Pytorch/RNN/2_TextClassification/FinalExam'

# nltk.download('stopwords')
# nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')

def cleanData(list_of_strings):
    cleaned_list = []
    for string in list_of_strings:
        tokens = nltk.word_tokenize(string)
        """ remove stopwords """
        tokens = [token.lower() for token in tokens if token.lower() not in stopwords]

        """ remove punctuation"""
        tokens = [token for token in tokens if token.isalnum()]

        """ lemmatize """
        wnl = nltk.WordNetLemmatizer()
        tokens = [wnl.lemmatize(token) for token in tokens]

        cleanstring = ' '.join(tokens)
        cleaned_list.append(cleanstring)
    return cleaned_list

train_data = pd.read_csv("Train.csv", encoding='latin-1')
trainData_list = train_data['Text'].tolist()
cleaned_trainData = cleanData(trainData_list)
train_data['cleaned_text'] = cleaned_trainData
train_data = train_data.drop(['Text'], axis = 1)
columns_titles = ["cleaned_text","Sentiment"]
train_data = train_data.reindex(columns=columns_titles)
# dict_map = {'Extremely Negative':0, 'Negative':1, 'Neutral':2, 'Positive':3, 'Extremely Positive':4}
# train_data = train_data.replace({"Sentiment": dict_map})

from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()
#
# Encode single column status
#
train_data.Sentiment = le.fit_transform(train_data.Sentiment)
#
# Print df.head for checking the transformation
#
# print(train_data.head())

""" removing rows with zero length strings """
data_list = train_data['cleaned_text'].tolist()
arr = []
for i in range(len(data_list)):
    string = data_list[i]
    tokens = nltk.word_tokenize(string)
    if len(tokens) == 0:
        arr.append(i)

print("Percentage of zero length sequences in train data, after data cleaning:", 100*len(arr)/len(train_data),"\n")
print("It is safe to discard these data points.")
train_data = train_data.drop(index = arr)

""" splitting train into train and validation dataframe """
random_seed = 42
train_data, val_data = train_test_split(train_data, train_size=0.9, random_state=random_seed)
train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)

test_data = pd.read_csv("Test_submission_atidke9.csv", encoding='latin-1')
testData_list = test_data['Text'].tolist()
cleaned_testData = cleanData(testData_list)
test_data['cleaned_text'] = cleaned_testData
test_data = test_data.drop(['Text'], axis = 1)
columns_titles = ["cleaned_text","Sentiment"]
test_data = test_data.reindex(columns=columns_titles)
# dict_map = {1:0, 2:1}
# test_data = test_data.replace({"Sentiment": dict_map})

""" removing rows with zero length strings """
data_list = test_data['cleaned_text'].tolist()
arr_test = []
for i in range(len(data_list)):
    string = data_list[i]
    tokens = nltk.word_tokenize(string)
    if len(tokens) == 0:
        arr_test.append(i)

print("Percentage of zero length sequences in test data, after data cleaning:", 100*len(arr_test)/len(test_data),"\n")

"""Setup"""
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""Import cleaned data"""

# #considering 25% of the train data to train the model, bigger model gives CUDA error with current GPU config
train_data, train_data2 = train_test_split(train_data, train_size=0.50, random_state=RANDOM_SEED)
train_data, train_data2 = train_data.reset_index(drop=True), train_data2.reset_index(drop=True)
#using train_data1 hereafter
#
# val_data = pd.read_csv('val_cleaned.csv')
# test_data = pd.read_csv('test_cleaned.csv')

class_names = [0,1,2,3,4]

"""Calculate max sequence length"""
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Concatenating all the dataframes
frames = [train_data, val_data]
df = pd.concat(frames)

token_lens = []
for txt in df.cleaned_text:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 512])
plt.xlabel('Token count')
plt.show()

#As can be seen from the graph, most of the reviews seem to contain less than 160 tokens, so we choose a maximum length of 400.

"""Creating Dataloader"""
MAX_LEN = 100 #cannot be greater than 512       # change this to a lower value

class GPReviewDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.reviews)
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.cleaned_text.to_numpy(),           # change text variable df.?.to_numpy()
    targets=df.Sentiment.to_numpy(),                  # same
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 32
#creatinf train dataloader with the 25% split created above
train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(val_data, tokenizer, MAX_LEN, BATCH_SIZE)
# test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

"""Sentiment classifier and helper functions"""
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim = 1)
  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(bert_output[1])
    output = self.out(output)
    return self.softmax(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)

EPOCHS = 2        # change
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch( model, data_loader, loss_fn,
  optimizer, device, scheduler, n_examples, epoch
):
  model = model.train()
  losses = []
  correct_predictions = 0
  loss_train, train_steps = 0, 0
  total = len(train_data) // BATCH_SIZE
  with tqdm(total=total, desc="Epoch {}".format(epoch)) as pbar:
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      loss_train += loss.item()
      losses.append(loss.item())
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      train_steps += 1
      pbar.update(1)
      pbar.set_postfix_str("Training Loss: {:.5f}".format(loss_train / train_steps))
  return correct_predictions / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

"""Training loop"""
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(train_data), epoch
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(val_data)
  )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc

"""Plotting train_acc, train_loss, val_acc, val_loss against the epochs"""
train_acc_hist_arr = torch.stack(history['train_acc']).cpu().numpy()
train_loss_hist_arr = np.array(history['train_loss'])
val_acc_hist_arr = torch.stack(history['val_acc']).cpu().numpy()
val_loss_hist_arr = np.array(history['val_loss'])

EpochHistDF = pd.DataFrame({'train_acc': train_acc_hist_arr, 'train_loss': train_loss_hist_arr, 'val_acc': val_acc_hist_arr,
                        'val_loss':val_loss_hist_arr}, columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
plt.plot(EpochHistDF['train_acc'], label='train accuracy')
plt.plot(EpochHistDF['train_loss'], label='train loss')
plt.plot(EpochHistDF['val_acc'], label='validation accuracy')
plt.plot(EpochHistDF['val_loss'], label='validation loss')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# """Testing the best model"""
# model.load_state_dict(torch.load("best_model_state.pt"))
# model.eval()
#
# test_acc, _ = eval_model(
#   model,
#   test_data_loader,
#   loss_fn,
#   device,
#   len(test_data)
# )
#
# print("Test accuracy for the best model:",test_acc.item())
# """"""

model.load_state_dict(torch.load("best_model_state.bin"))
# test_pred = model.predict(test_input)
# test_pred.shape

predicted_label = []
for i in range(len(test_data)):
    text = test_data['cleaned_text'][i]
    encoded_review = tokenizer.encode_plus( text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    predicted_label.append(prediction.cpu().item())
    
original_testdf = pd.read_csv('Test_submission_atidke9.csv', encoding='latin-1')
original_testdf['Sentiment'] = predicted_label
original_testdf.to_csv('Test_submission_atidke9.csv', index = False)
