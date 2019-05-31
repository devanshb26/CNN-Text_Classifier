import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import numpy as np

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype = torch.float)

fields = [(None,None),(None,None),('text', TEXT),('label', LABEL)]
train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '',
                                        train = 'V1.4_Training.csv',
                                        validation = 'SubtaskA_EvaluationData_labeled.csv',
                                        test = 'SubtaskA_Trial_Test_Labeled - Copy.csv',
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
                  
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size = BATCH_SIZE, 
    device = device)
                  
                  
                  
                  
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,hidden_dim,
                 dropout, pad_idx,Dropout_2):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, output_dim)
#         self.relu=nn.Relu()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
#         self.dropout_2=nn.Dropout(Dropout_2)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
#         print(conv.shape[2]+"   "+conv.shape[1] for conv in conved)        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
#         f1=self.fc1(cat)
#         f1_relu=F.relu(f1)
#         d=self.dropout_2(f1_relu)
#         return self.fc2(d)
        cat_relu=F.relu(cat)
        return self.fc1(cat_relu)
                 
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 150
HIDDEN_DIM=250
Dropout_2=0.75
FILTER_SIZES = [2,3]
OUTPUT_DIM = 1
DROPOUT = 0.6
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, HIDDEN_DIM,DROPOUT, PAD_IDX,Dropout_2)
                  
                  
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
    f1=f1_score((y.data).cpu().numpy(),(rounded_preds.data).cpu().numpy(),average='binary')
    y_mini=(y.data).cpu().numpy()
    pred_mini=(y.data).cpu().numpy()
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

          epoch_loss += loss.item()
          epoch_acc += acc.item()
          epoch_f1+=f1
          y_tot=np.concatenate([y_tot,y_mini])
          pred_tot=np.concatenate([pred_tot,pred_mini])
  f1=f1_score(y_tot,pred_tot,average='binary')
  f1_macro=f1_score(y_tot,pred_tot,average='macro')
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator),f1,f1_macro



import time

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 50
best_valid_f1 = float('inf')

for epoch in range(N_EPOCHS):

  start_time = time.time()

  train_loss, train_acc,train_f1 = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc,valid_f1,f1,f1_macro = evaluate(model, valid_iterator, criterion)

  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  if valid_f1 < best_valid_f1:
      best_valid_f1 = valid_f1
      torch.save(model.state_dict(), 'tut4-model.pt')

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%| Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%| Valid_f1 : {valid_f1:.4f}')



  model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')


import spacy
nlp = spacy.load('en')

def predict_sentiment(model, sentence, min_len = 5):
  model.eval()
  tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
  if len(tokenized) < min_len:
      tokenized += ['<pad>'] * (min_len - len(tokenized))
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(1)
  prediction = torch.sigmoid(model(tensor))
  return prediction.item()


