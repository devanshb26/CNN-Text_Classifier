import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm

import random
import re
from torch.backends import cudnn
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# seed = 0
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

URL_REGEX = "(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
EMAIL_REGEX = "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
USER_REGEX = "\\@\\w+"

INVISIBLE_REGEX = '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]'
QUOTATION_REGEX = "[”“❝„\"]+"
APOSTROPHE_REGEX = "[‘´’̇]+"

PRICE_REGEX = '([\$£€¥][0-9]+|[0-9]+[\$£€¥])'
DATE_REGEX = '(?:(?:(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
             '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,' \
             '4})\\b))|(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
             '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(' \
             '?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))'
TIME_REGEX = '(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)'

import spacy
nlp = spacy.load('en')
def tokenize_en(text):
  
  
  text = re.sub('[\u0370-\u03ff]', '', text)  # Greek and Coptic
  text = re.sub('[\u0400-\u052f]', '', text)  # Cyrillic and Cyrillic Supplementary
  text = re.sub('[\u2500-\u257f]', '', text)  # Box Drawing
  text = re.sub('[\u2e80-\u4dff]', '', text)  # from CJK Radicals Supplement
  text = re.sub('[\u4e00-\u9fff]', '', text)  # CJK Unified Ideographs
  text = re.sub('[\ue000-\uf8ff]', '', text)  # Private Use Area
  text = re.sub('[\uff00-\uffef]', '', text)  # Halfwidth and Fullwidth Forms
  text = re.sub('[\ufe30-\ufe4f]', '', text)  # CJK Compatibility Forms

  text = re.sub(INVISIBLE_REGEX, '', text)
  text = re.sub(QUOTATION_REGEX, '\"', text)
  text = re.sub(APOSTROPHE_REGEX, '\'', text)
  text = re.sub(r"\s+", " ", text)
  
  text = re.sub(PRICE_REGEX, r" <PRICE> ", text)
  text = re.sub(TIME_REGEX, r" <TIME> ", text)
  text = re.sub(DATE_REGEX, r" <DATE> ", text)

  text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
  text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
  text = re.sub(r" [0-9]+ ", r" <NUMBER> ", text)

  text = re.sub(r"(\b)([Ii]) 'm", r"\1\2 am", text)
  text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 're", r"\1\2 are", text)
  text = re.sub(r"(\b)([Ll]et) 's", r"\1\2 us", text)
  text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 'll", r"\1\2 will", text)
  text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou) 've", r"\1\2 have", text)
  
  text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould) n't", r"\1\2 not", text)
  text = re.sub(r"(\b)([Cc]a) n't", r"\1\2n not", text)
  text = re.sub(r"(\b)([Ww]) on't", r"\1\2ill not", text)
  text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
  text = re.sub(r" n't ", r" not ", text)

  
  text = re.sub(r"([‼.,;:?!…])+", r" \1 ", text)
  text = re.sub(r"([()])+", r" \1 ", text)
  text = re.sub(r"[-]+", r" - ", text)
  text = re.sub(r"[_]+", r" _ ", text)
  text = re.sub(r"[=]+", r" = ", text)
  text = re.sub(r"[\&]+", r" \& ", text)
  text = re.sub(r"[\+]+", r" \+ ", text)

  text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
  text = re.sub(r"what's", "what is", text)
  text = re.sub(r"\'s", "", text)
  text = re.sub(r"\'ve", "have", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"n't", "not", text)
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"\'re", "are", text)
  text = re.sub(r"\'d", "would", text)
  text = re.sub(r"\'ll", "will", text)
  text = re.sub(r",", "", text)
  text = re.sub(r"\.", "", text)
  text = re.sub(r"!", "!", text)
  text = re.sub(r"\/", "", text)
  text = re.sub(r"\^", "^", text)
  text = re.sub(r"\+", "+", text)
  text = re.sub(r"\-", "-", text)
  text = re.sub(r"\=", "=", text)
  text = re.sub(r"'", "", text)
  text = re.sub(r"<", "", text)
  text = re.sub(r">", "", text)
  text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
  text = re.sub(r":", ":", text)
  text = re.sub(r" e g ", "eg", text)
  text = re.sub(r" b g ", "bg", text)
  text = re.sub(r" u s ", "american", text)
  text = re.sub(r"\0s", "0", text)
  text = re.sub(r"e - mail", "email", text)
  text = re.sub(r"j k", "jk", text)
  tokenized=[tok.text for tok in nlp(text)]
  if len(tokenized) < 3:
        tokenized += ['<pad>'] * (3 - len(tokenized))
  return tokenized


TEXT = data.Field(tokenize=tokenize_en)
LABEL = data.LabelField(dtype = torch.float)

fields = [(None,None),(None,None),('text', TEXT),('label', LABEL)]
train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '',
                                        train = 'V1.4_Training.csv',
                                        validation = 'SubtaskB_Trial_Test_Labeled - Copy.csv',
                                        test = 'SubtaskB_EvaluationData_labeled.csv',
#                                         train = 'train_spacy.csv',
#                                         validation = 'valid_spacy.csv',
#                                         test = 'test_spacy.csv',
#                                         #sort_key=lambda x: len(x.Text),
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
)
print(vars(train_data[0]))
MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = 'glove.6B.100d', 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
                  
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size = BATCH_SIZE, 
    device = device)
                  
                  
                  
                  
class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx,dropout_2):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 364)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(364,162)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(162,50)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(50,output_dim)
        nn.init.kaiming_normal_(self.fc4.weight)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout_2)
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
        
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        embedded=self.dropout(embedded)
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = torch.cat(pooled, dim = 1)
        cat=self.dropout_2(cat)
        out=self.dropout_2(self.relu(self.fc1(cat)))
        out=self.dropout_2(self.relu(self.fc2(out)))
        out=self.relu(self.fc3(out))
#         out=self.dropout(out)
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc4(out)
                 
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 192
HIDDEN_DIM=250
Dropout_2=0.2
FILTER_SIZES = [2,3,4]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM,DROPOUT, PAD_IDX,Dropout_2)
                  
                  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')



pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


from sklearn.metrics import f1_score,confusion_matrix as cm
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    #print(len((y.data).cpu().numpy()))
#     f1=f1_score((y.data).cpu().numpy(),(rounded_preds.data).cpu().numpy(),average='binary')
    y_mini=(y.data).cpu().numpy()
    pred_mini=(rounded_preds.data).cpu().numpy()
    f1=f1_score(y_mini,pred_mini,average='binary')
    acc = correct.sum() / len(correct)
    return acc,f1,y_mini,pred_mini
  
  
def train(model, iterator, optimizer, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
 
  model.train()

  for batch in iterator:

      optimizer.zero_grad()
#       print(batch)
      predictions = model(batch.text).squeeze(1)

      loss = criterion(predictions, batch.label)

      acc,f1,y_mini,pred_mini= binary_accuracy(predictions, batch.label)
      #print(type(f1))
      loss.backward()

      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()
      epoch_f1=epoch_f1+f1
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator)



def evaluate(model, iterator, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
  y_tot=np.array([])
  pred_tot=np.array([])
  model.eval()

  with torch.no_grad():

      for batch in iterator:

          predictions = model(batch.text).squeeze(1)

          loss = criterion(predictions, batch.label)

          acc,f1,y_mini,pred_mini = binary_accuracy(predictions, batch.label)
#           print(cm(y_mini,pred_mini))
#           print(cr(y_mini,pred_mini))
          epoch_loss += loss.item()
          epoch_acc += acc.item()
          epoch_f1+=f1
          y_tot=np.concatenate([y_tot,y_mini])
          pred_tot=np.concatenate([pred_tot,pred_mini])
  f1=f1_score(y_tot,pred_tot,average='binary')
  f1_macro=f1_score(y_tot,pred_tot,average='macro')
  print(len(y_tot))
  print(cr(y_tot,pred_tot))
  print(cm(y_tot,pred_tot))
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator),f1,f1_macro



import time

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 20
best_valid_f1 = float(0)
c=0
for epoch in range(N_EPOCHS):

  start_time = time.time()

  train_loss, train_acc,train_f1 = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc,valid_f1,f1,f1_macro = evaluate(model, valid_iterator, criterion)

  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  if f1 > best_valid_f1:
      best_valid_f1 = f1
      torch.save(model.state_dict(), 'tut2-model.pt')
      c=0
  else:
    c=c+1
#   if c==3:
#     print(epoch)
#     break
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%| Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%| Valid_f1 : {f1:.4f}')


model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')


import spacy
nlp = spacy.load('en')

def predict_sentiment(model):
    model.eval()
    l=[]
    df=pd.read_csv("SubtaskA_Trial_Test_Labeled - Copy.csv")
    for i in range(len(df)):
      tokenized = tokenize_en(df['data'][i])
      indexed = [TEXT.vocab.stoi[t] for t in tokenized]
      length = [len(indexed)]
      tensor = torch.LongTensor(indexed).to(device)
      tensor = tensor.unsqueeze(1)
      length_tensor = torch.LongTensor(length)
      prediction = torch.sigmoid(model(tensor))
      l.append(prediction.item())
    df['preds']=l
    import csv
    df.to_csv('predidctions.csv')
    return(l)
    
    
a=predict_sentiment(model)

