import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
# URL_REGEX = "(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
# EMAIL_REGEX = "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
# USER_REGEX = "\\@\\w+"

# INVISIBLE_REGEX = '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]'
# QUOTATION_REGEX = "[”“❝„\"]+"
# APOSTROPHE_REGEX = "[‘´’̇]+"

# PRICE_REGEX = '([\$£€¥][0-9]+|[0-9]+[\$£€¥])'
# DATE_REGEX = '(?:(?:(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
#              '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,' \
#              '4})\\b))|(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
#              '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(' \
#              '?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))'
# TIME_REGEX = '(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)'

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
#   if len(tokenized) < 3:
#         tokenized += ['<pad>'] * (3 - len(tokenized))
  return tokenized


from sklearn.metrics import f1_score,classification_report as cr,confusion_matrix as cm
TEXT = data.Field(tokenize='spacy',include_lengths = True)
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
# batch_size changed from 64 to 16                  
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch = True,
    device = device)


import torch.nn as nn


# class Attention_Net(nn.Module):
#     def __init__(self):
#         super(Attention_Net, self).__init__()

class RNN(nn.Module):
# class Attention_Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx,dropout_2,batch_size):
		
        
        super().__init__()
#         super(Attention_Net, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_dim
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_length = embedding_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.relu=nn.ReLU()
#         self.attention_layer = Attention(hidden_dim * 2,128)
#         torch.nn.init.xavier_uniform(self.rnn.weight)
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        nn.init.kaiming_normal_(self.fc1.weight)
#         self.fc2 = nn.Linear(25,output_dim)
#         nn.init.kaiming_normal_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout_2)
#         for m in self.modules():
#           if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform(m.weight)
#             m.bias.data.fill_(0.0)

		
#     def attention_net(self, lstm_output, final_state):
# 	hidden = final_state.squeeze(0)
#         attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
#         soft_attn_weights = F.softmax(attn_weights, 1)
#         new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
# 	return new_hidden_state
    def attention(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
	print(lstm_output.size)
	print(hidden.unsqueeze(2).size)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
    def forward(self, text, text_lengths):
	
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        input = embedded.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        
        packed_output, (hidden, cell) = self.rnn(input)
	
#         packed_output, (hidden, cell) = self.rnn(packed_embedded,(h_0,c_0))
#         packed_output = packed_output.permute(1, 0, 2)
        #unpack sequence
#         output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        packed_output.permute(1, 0, 2)
        hidden = self.dropout_2(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        attn_output = self.attention(packed_output, hidden)      
#         hidden = [batch size, hid dim * num directions]
#         h_lstm_atten = self.attention_layer(hidden)
#         out = self.fc1(hidden.squeeze(0))
        out = self.fc1(attn_output)
#         out=self.relu(out)
        return out
        

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
# hidden_dim changed from 256 to 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
# dropout changed from 0.3 to 0.5
DROPOUT = 0.3
dropout_2=0.3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


        

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX,
           dropout_2,BATCH_SIZE)
            
          
            
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
 
 
def evaluate(model, iterator, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
  y_tot=np.array([])
  pred_tot=np.array([])
  model.eval()

  with torch.no_grad():

      for batch in iterator:
          text, text_lengths = batch.text
          predictions = model(text,text_lengths).squeeze(1)

          loss = criterion(predictions, batch.label)

          acc,f1,y_mini,pred_mini = binary_accuracy(predictions, batch.label)

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
      c=0
      torch.save(model.state_dict(), 'tut4-model.pt')
  else:
    c=c+1
  if c==6:
    print(epoch)
    break
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%| Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%| Valid_f1 : {valid_f1:.4f}')



  model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')   

# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """

#     #round predictions to the closest integer
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float() #convert into float for division 
#     acc = correct.sum() / len(correct)
#     return acc
  
# def train(model, iterator, optimizer, criterion):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.train()
    
#     for batch in iterator:
        
#         optimizer.zero_grad()
        
#         text, text_lengths = batch.text
        
#         predictions = model(text, text_lengths).squeeze(1)
        
#         loss = criterion(predictions, batch.label)
        
#         acc = binary_accuracy(predictions, batch.label)
        
#         loss.backward()
        
#         optimizer.step()
        
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
  
# def evaluate(model, iterator, criterion):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.eval()
    
#     with torch.no_grad():
    
#         for batch in iterator:

#             text, text_lengths = batch.text
            
#             predictions = model(text, text_lengths).squeeze(1)
#             print(text_lengths)
#             loss = criterion(predictions, batch.label)
            
#             acc = binary_accuracy(predictions, batch.label)
#             print(acc.item())
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
  
  
  
# import time

# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
  
# N_EPOCHS = 5

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):

#     start_time = time.time()
    
#     train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
#     valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
#     end_time = time.time()

#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'tut2-model.pt')
    
#     print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
# model.load_state_dict(torch.load('tut2-model.pt'))

# test_loss, test_acc = evaluate(model, test_iterator, criterion)

# print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

