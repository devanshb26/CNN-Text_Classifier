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
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import spacy
nlp = spacy.load('en')
def tokenize_en(text):
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



from sklearn.metrics import f1_score,classification_report as cr,confusion_matrix as cm
TEXT = data.Field(tokenize=tokenize_en,include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

fields = [(None,None),(None,None),('text', TEXT),('label', LABEL)]
train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '',
                                        train = 'V1.4_Training.csv',
                                        validation = 'SubtaskA_Trial_Test_Labeled - Copy.csv',
                                        test = 'SubtaskA_EvaluationData_labeled.csv',
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
# batch size changed from 64 to 16                  
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
#     shuffle=False,
    sort_key=lambda x: len(x.text),
    sort_within_batch = True,
    device = device)


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,  n_filters, filter_sizes,hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        for name, param in self.rnn.named_parameters():
         
         if 'bias' in name:
           nn.init.constant_(param, 0.0)
#            print(0)
#            print(param)
         elif 'weight' in name:
#            print(1)
           nn.init.kaiming_normal_(param)
           
        
        #self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        
        
        
        self.relu = nn.ReLU()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters+hidden_dim * 2, 128)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(64, output_dim)
        nn.init.kaiming_normal_(self.fc3.weight)
        
        
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        text_cnn = text.permute(1, 0)
        embedded = self.dropout(self.embedding(text))
        ###################
        embedded_cnn = self.embedding(text_cnn)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded_cnn = embedded_cnn.unsqueeze(1)
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #########
        conved = [F.relu(conv(embedded_cnn)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        ########
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        ##########
        cnn_x = torch.transpose(cat, 0, 1)
        bilstm_out = torch.transpose(hidden, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        lstm_cnn = torch.transpose(cnn_bilstm_out, 0, 1)
#         print(lstm_cnn.size())
#         lstm_cnn=torch.cat((cat,hidden.squeeze(0)),dim=1)
        #hidden = [batch size, hid dim * num directions]
        out=self.fc1(lstm_cnn)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
            
        return out
        

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
#no. of filters from 150 to 250 for sub_B
N_FILTERS = 150
FILTER_SIZES = [2,3]
N_LAYERS = 2
BIDIRECTIONAL = True
#dropout from 0.5 to 0.3 for sub_B
DROPOUT = 0.7
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM,
            N_FILTERS,
            FILTER_SIZES,
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)
            
            
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)


import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    #print(len((y.data).cpu().numpy()))
    f1=f1_score((y.data).cpu().numpy(),(rounded_preds.data).cpu().numpy(),average='binary')
    y_mini=(y.data).cpu().numpy()
    pred_mini=(rounded_preds.data).cpu().numpy()
    acc = correct.sum() / len(correct)
    preds=torch.sigmoid(preds).data.cpu().numpy()
    return acc,f1,y_mini,pred_mini
                  
def train(model, iterator, optimizer, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
 
  model.train()
  
  for batch in iterator:
      text, text_lengths = batch.text
      optimizer.zero_grad()
#       print(batch)
      predictions = model(text,text_lengths).squeeze(1)

      loss = criterion(predictions, batch.label)

      acc,f1,y_mini,pred_mini= binary_accuracy(predictions, batch.label)
      #print(type(f1))
      loss.backward()

      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()
      epoch_f1=epoch_f1+f1
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator)
 
# z=0 
def evaluate(model, iterator, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
  y_tot=np.array([])
  pred_tot=np.array([])
  model.eval()
  full_text=[]
  full_probs=np.array([])
  with torch.no_grad():

      for batch in iterator:
          text, text_lengths = batch.text
          predictions = model(text,text_lengths).squeeze(1)
#           print(type(text))
          loss = criterion(predictions, batch.label)

          acc,f1,y_mini,pred_mini = binary_accuracy(predictions, batch.label)
#           t=text.tolist()
#           print(t[0])
#           t=[' '.join(q) for q in t]
#           full_text+=t
#           full_probs=np.concatenate([full_probs,preds])
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
  
#   df=pd.DataFrame({'data':full_text,'probs':full_probs})
#   df.to_csv('probs_evaluation.csv')
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
      c=0
      torch.save(model.state_dict(), 'tut4-model.pt')
  else:
    c=c+1
#   if c==3:
#     print(epoch)
#     break
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%| Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%| Valid_f1 : {valid_f1:.4f}')


model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro = evaluate(model, test_iterator, criterion)


print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')  


def predict_sentiment(model):
    model.eval()
    l=[]
    df=pd.read_csv('SubtaskA_EvaluationData_labeled.csv')
    for i in range(len(df)):
      tokenized = tokenize_en(df['data'][i])
      indexed = [TEXT.vocab.stoi[t] for t in tokenized]
      length = [len(indexed)]
      tensor = torch.LongTensor(indexed).to(device)
      tensor = tensor.unsqueeze(1)
      length_tensor = torch.LongTensor(length)
      prediction = torch.sigmoid(model(tensor, length_tensor))
      l.append(prediction.item())
    df['preds']=l
    import csv
    df.to_csv('predidctions.csv')
    return(l)
    
    
a=predict_sentiment(model)
