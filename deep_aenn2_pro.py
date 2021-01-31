import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from IPython.display import Image, display
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")
USE_CUDA = True
#initialize random seed
seed_value = 1234
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
#convert dataset to dataframe
df = pd.read_csv("fraud_dataset_v1.csv")
#remove the categorical label column to prepare dataframe for preprocessing
label = df.pop('label')
#plot posting key and general ledger account 
#fig, ax = plt.subplots(1,2)
#fig.set_figwidth(20)
#plot the first for posting key attribute
#g = sns.countplot(x = df['BSCHL'], ax = ax[0])
#g.set_xticklabels(g.get_xticklabels())
#g.set_title('Distribution of posting key attribute values')
#plot general ledger account attribute
#g = sns.countplot(x = df['HKONT'], ax = ax[1])
#g.set_xticklabels(g.get_xticklabels())
#g.set_title("Distribtution of general ledger account attribute values")
#select categorical attributes for one-hot encoding
cat_var_trans = ['WAERS','BUKRS','KTOSL','BSCHL','HKONT']
df_cat_trans = pd.get_dummies(df[cat_var_trans])
#to inspect
df_cat_trans.head(10)
num_attr_names = ['DMBTR','WRBTR']
#plot local and document amount attribute
#g = sns.pairplot(data = df, vars = num_attr_names, hue = 'label')
#g.fig.suptitle("Distribution of DMBTR vs WRBTR amount values")
#g.fig.set_size_inches(15,5)
#amount values are heavy tailed so normalizing to reach global minima faster
#adding a small constant to eliminate the zero values for log scaling
num_attr = df[num_attr_names] + 1e-7
num_attr = num_attr.apply(np.log)
#normalize between [0,1]
df_num_attr = (num_attr-num_attr.min())/(num_attr.max()-num_attr.min())
#visualize and append label for category distinction
num_attr_vis = df_num_attr.copy()
num_attr_vis['label'] = label
#plot log scale graph
#g = sns.pairplot(data = num_attr_vis, vars = num_attr_names, hue = 'label')
#g.fig.suptitle("Distribution of DMBTR vs WRBTR amount values")
#g.fig.set_size_inches(15,5)
#joining the encoded and normalized attributes
df_trans = pd.concat([df_cat_trans, df_num_attr], axis = 1)
df_trans.shape

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        #layer1 in 618, out 512
        self.encoder_L1 = nn.Linear(in_features = df_trans.shape[1], out_features = 512, bias  = True)# add linearity
        nn.init.xavier_uniform_(self.encoder_L1.weight)# initialize weights
        self.encoder_R1 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)#add non-linearities leakyrelu to avoid dying non-linearities to speed up training convergence
        #layer2 in 512 out 256
        self.encoder_L2 = nn.Linear(in_features = 512, out_features = 256, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L2.weight)
        self.encoder_R2 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer3 in 256 out 128
        self.encoder_L3 = nn.Linear(in_features = 256, out_features = 128, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L3.weight)
        self.encoder_R3 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer4 in 128 out 64
        self.encoder_L4 = nn.Linear(in_features = 128, out_features = 64, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L4.weight)
        self.encoder_R4 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer5 in 64 out 32
        self.encoder_L5 = nn.Linear(in_features = 64, out_features = 32, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L5.weight)
        self.encoder_R5 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer6 in 32 out 16
        self.encoder_L6 = nn.Linear(in_features = 32, out_features = 16, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L6.weight)
        self.encoder_R6 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer7 in 16 out 8
        self.encoder_L7 = nn.Linear(in_features = 16, out_features = 8, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L7.weight)
        self.encoder_R7 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer8 in 8 out 4
        self.encoder_L8 = nn.Linear(in_features = 8, out_features = 4, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L8.weight)
        self.encoder_R8 = nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #layer9 in 4 out 3
        self.encoder_L9 = nn.Linear(in_features = 4, out_features = 3, bias  = True)
        nn.init.xavier_uniform_(self.encoder_L9.weight)
        self.encoder_R9= nn.LeakyReLU(negative_slope = 0.4, inplace = True)
        #dropout layer for probabilit = p
        self.dropout = nn.Dropout(p = 0.0, inplace = True)
        
    def forward(self, x):
        #network layers forward pass through
        x = self.encoder_R1(self.dropout(self.encoder_L1(x)))
        x = self.encoder_R2(self.dropout(self.encoder_L2(x)))
        x = self.encoder_R3(self.dropout(self.encoder_L3(x)))
        x = self.encoder_R4(self.dropout(self.encoder_L4(x)))
        x = self.encoder_R5(self.dropout(self.encoder_L5(x)))
        x = self.encoder_R6(self.dropout(self.encoder_L6(x)))
        x = self.encoder_R7(self.dropout(self.encoder_L7(x)))
        x = self.encoder_R8(self.dropout(self.encoder_L8(x)))
        x = self.encoder_R9(self.encoder_L9(x))
        
        return x
        
#initialize training
encoder_train = encoder()
print(encoder_train)#network summary

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder_L1 = nn.Linear(in_features=3, out_features=4, bias= True)
        nn.init.xavier_uniform_(self.decoder_L1.weight)
        self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L2 = nn.Linear(in_features=4, out_features=8, bias= True)
        nn.init.xavier_uniform_(self.decoder_L2.weight)
        self.decoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L3 = nn.Linear(in_features=8, out_features=16, bias= True)
        nn.init.xavier_uniform_(self.decoder_L3.weight)
        self.decoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L4 = nn.Linear(in_features=16, out_features=32, bias= True)
        nn.init.xavier_uniform_(self.decoder_L4.weight)
        self.decoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L5 = nn.Linear(in_features=32, out_features=64, bias= True)
        nn.init.xavier_uniform_(self.decoder_L5.weight)
        self.decoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L6 = nn.Linear(in_features=64, out_features=128, bias= True)
        nn.init.xavier_uniform_(self.decoder_L6.weight)
        self.decoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L7 = nn.Linear(in_features=128, out_features=256, bias= True)
        nn.init.xavier_uniform_(self.decoder_L7.weight)
        self.decoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L8 = nn.Linear(in_features=256, out_features=512, bias= True)
        nn.init.xavier_uniform_(self.decoder_L8.weight)
        self.decoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        self.decoder_L9 = nn.Linear(in_features= 512, out_features=df_trans.shape[1], bias= True)
        nn.init.xavier_uniform_(self.decoder_L9.weight)
        self.decoder_R9 = nn.LeakyReLU(negative_slope=0.4, inplace= True)
        
        self.dropout = nn.Dropout(p=0.0, inplace= True)
        
    def forward(self, x):
        x = self.decoder_R1(self.dropout(self.decoder_L1(x)))
        x = self.decoder_R2(self.dropout(self.decoder_L2(x)))
        x = self.decoder_R3(self.dropout(self.decoder_L3(x)))
        x = self.decoder_R4(self.dropout(self.decoder_L4(x)))
        x = self.decoder_R5(self.dropout(self.decoder_L5(x)))
        x = self.decoder_R6(self.dropout(self.decoder_L6(x)))
        x = self.decoder_R7(self.dropout(self.decoder_L7(x)))
        x = self.decoder_R8(self.dropout(self.decoder_L8(x)))
        x = self.decoder_R9(self.decoder_L9(x))
        
        return x

decoder_train = decoder()
print(decoder_train)
loss_function = nn.BCEWithLogitsLoss(reduction = 'mean')
learning_rate = 1e-3
encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr = learning_rate)
decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr = learning_rate)

#training process
num_epochs = 5
mini_batch_size = 128
torch_dataset = torch.from_numpy(df_trans.values).float()
dataloader = DataLoader(torch_dataset.cuda(), batch_size=mini_batch_size, shuffle= True)
losses= []
data = autograd.Variable(torch_dataset)# encoded data to torch variable
for epoch in range(num_epochs):
    mini_batch_counter = 0
    if(torch.backends.cudnn.version()!= None) and (USE_CUDA == True):
        encoder_train.cuda()
        decoder_train.cuda()
    encoder_train.train()
    decoder_train.train()
    start_time = datetime.now()
    
    #iterate over all batches
    for mini_batch_data in dataloader:
        mini_batch_counter += 1
        mini_batch_torch = autograd.Variable(mini_batch_data)
        
        #1. run forward pass
        z_rep = encoder_train(mini_batch_torch)
        mini_batch_reconstruction = decoder_train(z_rep)
        
        #2. reconstruction loss
        reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_torch)
        
        #3. backward pass
        #reset gradients
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        #run backward pass
        reconstruction_loss.backward()
        
        #4. update model parameters
        decoder_optimizer.step()
        encoder_optimizer.step()
        
        # monitor training process
        if mini_batch_counter % 1000 == 0:
            mode = 'GPU'
            now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            end_time = datetime.now()-start_time
            print('[LOG{}], training status, epoch: [{:04}/{:04}],batch: {:04}, loss: {}, mode: {}, time required: {}' .format(now, (epoch+1), num_epochs, mini_batch_counter, np.round(reconstruction_loss.item(),4), mode, end_time))
            start_time = datetime.now()
    #!!!!evaluate model performance on training data *use when changing epochs !!! change the address according to machine 
    #encoder_train.cpu().eval()# on CPU to check for entire dataset GPU less memory
    #decoder_train.cpu().eval()
    #encoder_train.eval()
    #decoder_train.eval()
    #reconstruct the encoded data
    #reconstruction = decoder_train(encoder_train(data))
    #reconstruction loss_all entries
    #reconstruction_loss_all = loss_function(reconstruction, data)
    #losses.extend([reconstruction_loss_all.item()])
    #now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    #print('[LOG{}] training status, epoch: [{:04}/{:04}], loss{:.10f}' .format(now, (epoch+1), num_epochs, reconstruction_loss_all.item()))
    #save checkpoint for model after 1 epoch
    #encoder_model_name= "ep_{}_encoder_model.pth".format((epoch+1))
    #torch.save(encoder_train.state_dict(), os.path.join("C:/Users/acer/models2", encoder_model_name))
    #decoder_model_name= "ep_{}_decoder_model.pth".format((epoch+1))
    #torch.save(decoder_train.state_dict(), os.path.join("C:/Users/acer/models2", decoder_model_name))        
#plt.plot(range(0, len(losses)),losses)
#plt.xlabel('Training epoch')
#plt.xlim([0, len(losses)])
#plt.ylabel("reconstruction error")
#plt.title("AENN training performance")
data = autograd.Variable(torch_dataset)#convert tensor to torch variable
#load latest trained model
encoder_model_name = "ep_5_encoder_model.pth"
decoder_model_name = "ep_5_decoder_model.pth"
encoder_eval = encoder()
decoder_eval = decoder()
encoder_eval.load_state_dict(torch.load(os.path.join("C:/Users/acer/models2", encoder_model_name)))
decoder_eval.load_state_dict(torch.load(os.path.join("C:/Users/acer/models2", decoder_model_name)))
encoder_eval.eval()
decoder_eval.eval()
reconstruction = decoder_eval(encoder_eval(data))
reconstruction_loss_transaction= np.zeros(reconstruction.size()[0])#initialize BCE error
for i in range(0, reconstruction.size()[0]):
    reconstruction_loss_transaction[i] = loss_function(reconstruction[i], data[i]).item()
    if(i% 10000 == 0):
        print("collected individual reconstruction loss of: {:06}/{:06} transactions" .format(i, reconstruction.size()[0]))
    
fig = plt.figure()
ax = fig.add_subplot(111)
plot_data = np.column_stack((np.arange(len(reconstruction_loss_transaction)), reconstruction_loss_transaction))#assign unique id to transactions
regular_data = plot_data[label == 'regular']
global_outliers = plot_data[label == 'global']
local_outliers = plot_data[label == 'local']
#plot reconstruction error scatter graph
ax.scatter(regular_data[:,0]*10,regular_data[:,1], c = 'C0', alpha = 0.4, marker = 'o', label = 'regular')
ax.scatter(global_outliers[:,0],global_outliers[:,1], c= 'C1', alpha = 0.4, marker = '^', label = 'global')
ax.scatter(local_outliers[:,0],local_outliers[:,1], c = 'C2', alpha = 0.4, marker = '*', label = 'local')
ax.legend(loc = 'best')
df['label'] = label
df[reconstruction_loss_transaction >= 0.1]
df.head()






