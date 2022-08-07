#These libraries help to interact with the operating system and the runtime environment respectively
import os
import sys

#Model/Training related libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Dataloader libraries
from torch.utils.data import DataLoader, Dataset

# empty nvidia cache for tensors to live on
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


##############################################################################
## Download the dataset at https://www.kaggle.com/c/idl-fall2021-hw1p2/data ##
##############################################################################



####################
## Training Phase ##
####################




##############################################################
## Custom MLP dataset to pad the spectrogram with 0 vectors ##
##############################################################
class MLPDataset(Dataset):
  def __init__(self, data, labels, context=0):
    # concatenate?
    self.X = data # (N, ) of 2-d numpy arrays (m, d)
    self.Y = labels # (N, ) of 1-d numpy arrays (m, )
    assert(len(self.X)==len(self.Y))
    self.context = context
    self.offset = context

    ### Define data index mapping (4-6 lines)
    self.index_map = []
    for i, x in enumerate(self.Y):
        for j, xx in enumerate(x):
            index_pair_X = (i, j)
            self.index_map.append(index_pair_X)

    ### Zero pad data as-needed for context size = 1 (1-2 lines)
    for i, x in enumerate(self.X):
        self.X[i] = np.pad(x, ((context, context), (0, 0)),
                            'constant', constant_values=0)
    self.length = len(self.index_map)

  def __len__(self):
    return self.length

  def __getitem__(self,index):

    ### Get index pair from index map (1-2 lines)
    i, j = self.index_map[index]

    ### Calculate starting timestep using offset and context (1 line)
    start_j = j + self.offset - self.context

    ## Calculate ending timestep using offset and context (1 line)
    end_j = j + self.offset + self.context + 1

    ### Get data at index pair with context (1 line)
    xx = np.concatenate(self.X[i][start_j:end_j, :])

    ### Get label at index pair (1 line)
    yy = self.Y[i][j]

    ### Return data at index pair with context and label at index pair (1 line)
    return xx, yy

  def collate_fn(batch):

      ### Select all data from batch (1 line)

      batch_x = np.array([x for x,y in batch])

      ### Select all labels from batch (1 line)
      batch_y = np.array([y for x,y in batch])

      ### Convert batched data and labels to tensors (2 lines)
      batch_x = torch.from_numpy(batch_x).to(device)
      batch_y = torch.from_numpy(batch_y).type(torch.int64).to(device) # for 71 phoneme labels, and convert to long

      ### Return batched data and labels (1 line)
      return batch_x, batch_y

###################
## Load the Data ##
###################

print('Loading data')
train_data_filepath = np.load("train.npy",allow_pickle=True)
train_label_filepath = np.load("train_labels.npy",allow_pickle=True)
val_data_filepath = np.load("dev.npy",allow_pickle=True)
val_label_filepath = np.load("dev_labels.npy",allow_pickle=True)

######################
## hyper parameters ##
######################

batch_size = 3194
context = 20

#################
## Dataloaders ##
#################

print('Loading training dataloader')
train_data = MLPDataset(train_data_filepath, train_label_filepath, context=context)
train_args = dict(shuffle = True, batch_size = batch_size, num_workers=0, collate_fn=MLPDataset.collate_fn, drop_last=True)
train_loader = DataLoader(train_data, **train_args)

print('Loading validation dataloader')
val_data = MLPDataset(val_data_filepath, val_label_filepath, context=context)
val_args = dict(shuffle = False, batch_size = batch_size, num_workers=0, collate_fn=MLPDataset.collate_fn, drop_last=True)
val_loader = DataLoader(val_data, **val_args)


###################################
## Model Architecture definition ##
###################################

class MLP(nn.Module):

    # define model elements
    def __init__(self):
        super(MLP, self).__init__()
        in_feature = 40 * (1 + 2*context) # number of frequency bins
        out_feature = 71 # number of phonemes

        # linear -> batchnorm -> activation -> dropout 
        # ^ repeated 4x
        self.model = nn.Sequential(
            nn.Linear(in_feature, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(), # layer 1
            nn.Dropout(p=.2),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(), # layer 2
            nn.Dropout(p=.2),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(), # layer 3
            nn.Dropout(p=.2),
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Sigmoid(), # layer 4
            nn.Linear(512, out_feature, bias=True)
          )

    def forward(self, x):
        return self.model.forward(x)


print('Creating model and sending to GPU')
model = MLP()
model.double() # double coeffcients (rather than floats)
model.to(device) # move model to GPU
print(model)

#####################################
## Define Criterion/ Loss function ##
#####################################

criterion = nn.CrossEntropyLoss()



###########################
## Define Adam Optimizer ##
###########################

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='min')



#####################
## Train the model ##
#####################

def train_model(train_loader, model):
    training_loss = 0
    
    # Set model in 'Training mode', computes batchnorm running mean/var
    model.train()
    
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_loader):
        
        optimizer.zero_grad()

        out = model(inputs)
        loss = criterion(out, targets)
        
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    training_loss /= len(train_loader)
    return training_loss



#######################################################
## number of predictions correct / total predictions ##
#######################################################

def accuracy_score(predictions, labels):
  assert(len(predictions) == len(labels))
  return (predictions == labels).sum() / len(np.concatenate(predictions))



########################
## Evaluate the model ##
########################

def evaluate_model(val_loader, model):
    
    predictions = []
    actuals = []
    
    # Set model in validation mode
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            
            out = model(inputs)
            loss = criterion(out, targets)
            
            out = out.cpu()
            out = out.detach().numpy() # 1d of dint
            actual = targets # dint 

            out = np.argmax(out, axis=1) # find the highest prob
            
            predictions.append(out)
            actuals.append(actual.cpu())
    
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # Calculate validation accuracy
    acc = accuracy_score(actuals, predictions)
    return acc, loss.item()


epochs = 20

print('Training with epochs=' + str(epochs))

for epoch in range(epochs):
    
    # Train
    training_loss = train_model(train_loader, model)

    # Validation
    val_acc, val_loss = evaluate_model(val_loader, model)

    # scheduler 
    scheduler.step(val_loss)

    # Print log of accuracy and loss
    print("Epoch: "+str(epoch)+", Training loss: "+str(training_loss)+", Validation loss:"+str(val_loss)+
          ", Validation accuracy:"+str(val_acc*100)+"%")

    # save model each epoch
    torch.save(model.state_dict(), f'./model{epoch}.txt')

torch.save(model.state_dict(), f'./model100.txt')



###################
## Testing Phase ##
###################

class TestDataset(Dataset):
    def __init__(self, data, context=0): 
        self.X = data # (N, ) of 2-d numpy arrays (m, d)
        self.context = context
        self.offset = context

        self.index_map = []
        for i, x in enumerate(data):
            for j, xx in enumerate(x):
                index_pair_X = (i, j)
                self.index_map.append(index_pair_X)

        for i, x in enumerate(self.X):
            self.X[i] = np.pad(x, ((context, context), (0, 0)), 
                                'constant', constant_values=0)    
        self.length = len(self.index_map)  

    def __len__(self):
        return self.length

    def __getitem__(self,index):

        i, j = self.index_map[index]
        start_j = j + self.offset - self.context
        end_j = j + self.offset + self.context + 1

        xx = np.concatenate(self.X[i][start_j:end_j, :])

        return torch.from_numpy(xx).to(device)


test_dataset = TestDataset(test, context=context)
test_args = dict(shuffle = False, batch_size = batch_size, num_workers=0, drop_last=False)
test_loader = DataLoader(test_dataset, **test_args)

test_predictions = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        out = model(inputs)
        label = torch.argmax(out, axis=1)
        test_predictions.append(label.cpu())

test_predictions = np.concatenate(test_predictions)

print('Number of predictions: ' + str(len(test_predictions)))


###################################
## Save output to submission.csv ##
###################################

zipped = list(zip(range(len(test_predictions)), test_predictions))
np.savetxt('submission.csv', zipped, fmt='%i,%i', header='id,label',comments='')



