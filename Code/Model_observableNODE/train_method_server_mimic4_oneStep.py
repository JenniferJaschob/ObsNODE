import numpy as np
import torch
import torch.nn as nn
import os 

os.chdir('/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/torchdiffeq/')
from adjoint_method.adjoint import odeint_adjoint

#import os
#from tqdm import tqdm
#import torch.nn.functional as Functorch
#from torch.utils.data import  DataLoader
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

###### plot
import torchmetrics
import matplotlib.pyplot as plt
import pickle
  
###############################################################################
### data process ###
###############################################################################
def float_data(data):
    t_data_all,x_data,y_data = data
    return (t_data_all.to(torch.float), x_data.to(torch.float), y_data.to(torch.float))


def last_index(data_all):
    if data_all.dim() == 2:
        data_all = data_all.unsqueeze(-1)
        
    last_valid_values_all = data_all[:,-1,:]
    for i in range(data_all.size(-1)):
        data = data_all[...,i]

        mask = ~torch.isnan(data) 
        last_valid_idx = data.shape[1] - 1 - mask.flip(dims=[1]).int().argmax(dim=1)
        last_valid_values = data[torch.arange(data.size(0)), last_valid_idx]
        last_valid_values[~mask.any(dim=1)] = float('nan')
        last_valid_values_all[:,i] = last_valid_values
        
    return last_valid_values_all

#merged_dataset    
def def_x_obs(x_obs,time_obs=None):
    if x_obs.dim() == 2:
        x_obs = x_obs.unsqueeze(-1) 
    if time_obs != None:
        if time_obs.dim() == 2:
            time_obs = time_obs.unsqueeze(-1) 
        x_obs = torch.cat([x_obs,time_obs],dim=-1)

    return x_obs

############################################################################### 
## method ###
############################################################################### 
class MyObservableNeuralODE_withNNs(nn.Module):
    def __init__(self, helperNN_list=None,activation = 'relu',hidden_sizes=32,num_layers=3,n_x=1,n_z=2,u=None,n_u=0):   
        super( MyObservableNeuralODE_withNNs, self).__init__()
        
        self.helperNN_list = helperNN_list
        self.u = u
        self.n_u = n_u
        self.n_x = n_x
        self.n_z = n_z
        self.input_size = self.n_z*self.n_x+self.n_u 
        self.output_size = self.n_x
        self.hidden_sizes = hidden_sizes
        self.num_layers=num_layers
        
        if activation == 'leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation =='tanh':
            self.activation=nn.Tanh()
        elif activation =='relu':
            self.activation=nn.ReLU()
        elif activation =='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation =='identity':
            self.activation=nn.Identity()
            
        layers = []
        self.layers_sizes= [self.input_size]+([self.hidden_sizes] * self.num_layers)+[self.output_size]
    
        for i in range(len(self.layers_sizes)-1):
            if i == len(self.layers_sizes)-2:
                layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))
            else:
                layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))
                layers.append(self.activation)

        self.net = nn.Sequential(*layers)

    def forward(self,t,z):
        
        u = self.u
        n_x = self.n_x
        n_z = self.n_z
        
        device = z.device
        
        if(z.dim()==1):
            z=z.unsqueeze(0)
        
        if self.helperNN_list == None:
            z_n_x_to_end  = z[...,self.n_x:]
        else:
                 
            z_n_x_to_end = []
            
            for j in range(1,n_z):
                z_nn = z[...,:n_x*j]
                z_i = z[...,n_x*j:n_x*j+n_x]
    
                helperNN = self.helperNN_list[j-1]
                
                if u != None:
                    if (u.dim()==1):
                           u=u.unsqueeze(0)
                    if (u.dim()==2):    
                           u=u.unsqueeze(-1)
                            
                    if t < u.size(1) - 1:
                        u_in = u[:, int(t),:]
                    else:
                        u_in = u[:, -1,:] 
                    helperNN.u = u_in 
                z_n_x_to_end.append(z_i+helperNN(t,z_nn))

        z_n_x_to_end = torch.cat(z_n_x_to_end, dim=1)
        if u != None:
            out = self.net(torch.cat([z,u_in],dim=-1))
        else:    
            out = self.net(z)    
        output = torch.cat([z_n_x_to_end.to(device) , out.to(device)],dim=-1)
        return  output
      
        
class MyhelperNN(nn.Module):
    def __init__(self, activation = 'relu',hidden_sizes=32,num_layers=3,n_z_i=10,u=None,n_u=0,output_size=8):
        super(MyhelperNN, self).__init__()
        
        self.input_size = n_z_i+n_u
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.u = u
        self.n_u = n_u
        self.n_z_i = n_z_i
        
        
        if activation == 'leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation =='tanh':
            self.activation=nn.Tanh()
        elif activation =='relu':
            self.activation=nn.ReLU()
        elif activation =='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation =='identity':
            self.activation=nn.Identity()
            
        layers = []

        self.layers_sizes=[self.input_size]+([self.hidden_sizes] * self.num_layers)+[self.output_size]
    
        for i in range(len(self.layers_sizes)-1):
            if i == len(self.layers_sizes)-2:
                layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))
            else:
                layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))
                layers.append(self.activation)

        
        self.net = nn.Sequential(*layers)
        
    def forward(self,t,z_i):
        if self.u != None:
            return self.net(torch.cat([z_i,self.u],dim=-1))

        else:
            return self.net(z_i)        
        
def initialize_imputation(X, W):
    # Assumes X, W are torch tensors on the same device
    W_A = W.sum(dim=0)  # sum over time dimension
    A = (X * W).sum(dim=0)

    A_mask = W_A > 0
    A[A_mask] = A[A_mask] / W_A[A_mask]

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if W_A[i, j] == 0:
                # fallback: avg over all time steps and samples for this variable
                A[i, j] = X[:, :, j].sum() / W[:, :, j].sum()
                W_A[i, j] = 1.0

    # final fallback: average over all values
    A[W_A == 0] = X[W == 1].mean()

    return A


class Vaderlayer(nn.Module):
    def __init__(self, A_init):
        super(Vaderlayer, self).__init__()
        self.weights = nn.Parameter(A_init)
    
    def forward(self, X, W):
        return (1-W) * self.weights + X*W
    
class LSTMRecognitionModel_with_nan(nn.Module):
    def __init__(self, n_z=2,n_x=1,n_t=1, hidden_dim=32, num_layers=1, padding_value=float('nan'),W_train=None,dropout=0.0):
        super(LSTMRecognitionModel_with_nan, self).__init__()
        self.input_size = n_x+n_t 
        self.output_size = (n_z-1)*n_x
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.n_t = n_t
        self.n_x = n_x
        self.n_z = n_z
        self.W_train = W_train
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        
        self.fc = nn.Linear(self.hidden_dim, 
                            self.output_size)

    def forward(self, x_obs):
        device = x_obs.device
        
        if x_obs.dim() == 2:
            x_obs = x_obs.unsqueeze(-1)
            
        h_0 = torch.zeros(self.num_layers, x_obs.size(0), self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers,x_obs.size(0), self.hidden_dim, device=device)

        W = (~torch.isnan(x_obs)).to(torch.float).to(device)

        if not torch.all(W == 1.0) and torch.all((W == 0.0) | (W == 1.0)):
            if self.n_t == 0:
                x_obs_t_s = last_index(x_obs)
            else:
                x_obs_t_s = last_index(x_obs[..., :self.n_t])
            
            nan_mask = torch.isnan(x_obs)
            X = x_obs.clone().detach()
            X[nan_mask] = 0.0
            A_init = initialize_imputation(X, W)
            vad = Vaderlayer(A_init).to(device)
            X = vad.forward(X,W)
            x_obs[nan_mask] = X[nan_mask]

        else:
            if self.n_t == 0:
                x_obs_t_s = x_obs[:,-1,:]
            else:
                x_obs_t_s = x_obs[:,-1, :self.n_t]     
                
        out_lstm , (h_n, c_n) = self.lstm(x_obs, (h_0, c_0))
        latent_output = self.fc(out_lstm[:,0,:])
        
        out = torch.cat([x_obs_t_s, latent_output.to(device)], dim=-1) 

        return out    
   
class MyModel_adjoint(nn.Module): 
    def __init__(self, Node, Observer):
        super(MyModel_adjoint, self).__init__()
        self.Node = Node
        self.Observer = Observer
        
    def forward(self, time, x_obs,method,options=None):
        device = x_obs.device

        z_now = self.Observer(x_obs).to(device)
        z_pred = odeint_adjoint(func=self.Node, y0=z_now, t=time, method=method,adjoint_params=self.parameters(),options=options)
        return z_pred.to(device)    
    
def MyOptimizer(model_parameters,optimizer_func='Adam',learning_rate=1e-3):
    if optimizer_func == 'Adadelta':    
        optimizer = torch.optim.Adadelta(model_parameters, lr=learning_rate)
    elif optimizer_func == 'Adagrad':    
        optimizer = torch.optim.Adagrad(model_parameters, lr=learning_rate)     
    elif optimizer_func == 'Adam':   
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_func == 'AdamW':    
        optimizer = torch.optim.AdamW(model_parameters, lr=learning_rate)        
    elif optimizer_func == 'SparseAdam':    
        optimizer = torch.optim.SparseAdam(model_parameters, lr=learning_rate)        
    elif optimizer_func == 'Adamax':    
        optimizer = torch.optim.SparseAdamax(model_parameters, lr=learning_rate)        
    elif optimizer_func == 'ASGD':
        optimizer = torch.optim.ASGD(model_parameters, lr=learning_rate)
    elif optimizer_func ==  'LBFGS':   
        optimizer = torch.optim.LBFGS(model_parameters, lr=learning_rate)
    elif optimizer_func == 'NAdam':   
        optimizer = torch.optim.NAdam(model_parameters, lr=learning_rate)
    elif optimizer_func == 'RAdam':   
        optimizer = torch.optim.RAdam(model_parameters, lr=learning_rate)
    elif optimizer_func == 'RMSprop':   
        optimizer = torch.optim.RMSprop(model_parameters, lr=learning_rate)
    elif optimizer_func == 'SGG':
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate)
    return optimizer  

def loss_function_1D(x_true, t_true, x_pred, t_pred,loss_op='default'):
    device = x_true.device
    seq_len, batch_size = x_true.size()
    loss_batch = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
 
        t_true_seq = t_true[:, i]
        x_true_seq = x_true[:, i]
        x_pred_seq = x_pred[:, i]
        
        valid_mask =  ~torch.isnan(x_true_seq)
        
        t_true_seq_valid = t_true_seq[valid_mask]
        x_true_seq_valid = x_true_seq[valid_mask]

        x_pred_aligned = x_pred_seq[valid_mask]

        if loss_op=='default':
            if t_true_seq_valid.size(0) != 0:
                numerator = (t_true_seq_valid[-1] - t_true_seq_valid[0]) * torch.trapezoid(torch.square(x_pred_aligned - x_true_seq_valid), t_true_seq_valid, dim=0)
                denominator =   torch.trapezoid(torch.square(x_true_seq_valid), t_true_seq_valid, dim=0)
                loss_batch[i] = numerator / denominator  

            else:
                loss_batch[i] = float('nan')

        elif loss_op=='mean':
            loss_batch[i] = torch.nanmean(torch.square(x_pred_aligned - x_true_seq_valid))
            
        elif loss_op == 'map':
            loss_batch[i] = torch.nanmean(torch.abs((x_pred_aligned - x_true_seq_valid) / x_true_seq_valid))
        elif loss_op == 'rmse':
            loss_batch[i] = torch.sqrt(torch.nanmean(torch.square(x_pred_aligned - x_true_seq_valid)))
            
    out = torch.nanmean(loss_batch).to(device)

    return out


def loss_function_sum(x_true, t_true, x_pred, t_pred,loss_op='default',n_x_dach = None,w= 0):
    device = x_true.device
    
    if n_x_dach == None:
        n_x_dach = x_true.size(-1)
        
    if w != 0:
        loss_train = torch.empty(x_true.size(-1)).to(device)
    else:
        loss_train = torch.empty(n_x_dach).to(device)
        
    #determine Loss over output values
    for i in range(n_x_dach):
        x_true_i = x_true[...,i]
        t_true_i = t_true[...,i]
        x_pred_i = x_pred[...,i]
        
        loss_i = loss_function_1D(x_true_i, t_true_i, x_pred_i, t_pred,loss_op=loss_op)
        loss_train[i] = loss_i
        
    #if dim output values smaller than all values of x than add Loss from other values with witht w (w can be 0)    
    if n_x_dach < x_true.size(-1):
        if w != 0:
            for i in range(n_x_dach,x_true.size(-1)):
                x_true_i = x_true[...,i]
                t_true_i = t_true[...,i]
                x_pred_i = x_pred[...,i]        
                loss_i = loss_function_1D(x_true_i, t_true_i, x_pred_i, t_pred,loss_op=loss_op)*w
                loss_train[i] = loss_i
                
    out = torch.nanmean(loss_train).to(device)

    return out



def handle_early_stopping(val_loss, best_val_loss_early_stopping, patience, early_stopping_counter, min_delta):

    if val_loss < best_val_loss_early_stopping - min_delta:
        best_val_loss_early_stopping = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            return True, best_val_loss_early_stopping, early_stopping_counter
    return False, best_val_loss_early_stopping, early_stopping_counter

def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)  
                
                
def data_time_step(x,step_size):

    T, B, F = x.shape
    x_expanded = torch.full((int(T/step_size), B, F), float('nan'))
    x_expanded[::int(1/step_size)] = x     
    
    return x_expanded           
                    
###############################################################################
def train_diff_ts(model_node,model_observer,x_train,x_val,t_train,t_val,u_train=None,u_val=None,x_train_static=None,x_val_static=None, time_obs_pred=False,
          list_index_t_s=[10],method='euler',options = None,optimizer_func='Adam',learning_rate=1e-3,epochs=20,batch_size = 32,
          min_delta=0.005,patience=10,step_size=1.0,
          save_results=False,save_path='models/save_results_default_name.pth', time_multidim = None,loss_op='default', hyperopt = False,
          n_x_dach = None,w= 0,data_step = False,step_pred=False,num_pred=2):#,thresh=None
    device = x_train.device
    

    ## define model ###########################################################
    model = MyModel_adjoint(Node = model_node, Observer= model_observer).to(device)
        
    ## define optimizer #######################################################
    optimizer = MyOptimizer(model_parameters=model.parameters(),optimizer_func=optimizer_func,learning_rate=learning_rate)

    
    ## define default loss ####################################################
    losses_train, losses_val = [], []
    best_val_loss, best_val_loss_early_stopping, early_stopping_counter = float("inf"), float("inf"), 0
    
    ## Training loop
    for epoch in range(1, epochs+1):
        model.train()
        if hyperopt == False:
            print(f"Epoch {epoch}/{epochs}")
        
        ## in batch ###########################################################
        random_indice_sequence=np.random.choice(np.arange(x_train.shape[1]), x_train.shape[1], replace=False)
        
        for iteration in range(int(np.floor(x_train.shape[1]/batch_size))):

            index = torch.tensor(random_indice_sequence[iteration*batch_size:(iteration+1)*batch_size],device = device)
            #with diffrent t_s
            loss_train_list = []
            
            #####
            for t_s_i in range(len(list_index_t_s)):
                index_t_s = torch.tensor(list_index_t_s[t_s_i],device = device)
            
                    
                if step_pred == False:
                            
                    x_train_before_batch, x_train_after_batch    = x_train[:index_t_s+1,index], x_train[index_t_s:,index]
                    t_train_before_batch, t_train_after_batch    = t_train[:index_t_s+1,index], t_train[index_t_s:,index]
                            
                    if u_train != None:
                        u_train_before_batch, u_train_after_batch = u_train[:index_t_s+1,index], u_train[index_t_s:,index]
                        model.Node.u = u_train_before_batch.transpose(0, 1)
                            
                    x_obs = x_train_before_batch.transpose(0, 1)
                        
                    #cat x_obs with time
                    if time_obs_pred == True:
                        x_obs = def_x_obs(x_obs=x_obs,time_obs=t_train_before_batch.transpose(0, 1)).to(device)
                            
                    max_t = int(torch.max(t_train_after_batch[~torch.isnan(t_train_after_batch)]))
                    t_pred_train_batch = torch.arange(int(index_t_s), max_t+1, step=step_size, device=device)
    
                    z_pred_batch  = model(t_pred_train_batch, x_obs,method,options=options).to(device)
    
                    if x_train_after_batch.dim()>2:
                        x_pred_batch = z_pred_batch[...,:x_train_after_batch.size(-1)]
                    else:
                        x_pred_batch = z_pred_batch[...,0]
                    
                    if data_step: x_train_after_batch = data_time_step(x_train_after_batch,step_size)
                    if data_step: t_train_after_batch = data_time_step(t_train_after_batch,step_size)
                    
                    if x_pred_batch.dim()==2:
                        loss_train_t_s = loss_function_1D(x_train_after_batch, t_train_after_batch, x_pred_batch, t_pred_train_batch,loss_op)
                    else:
                        loss_train_t_s = loss_function_sum(x_train_after_batch, t_train_after_batch, x_pred_batch, t_pred_train_batch,loss_op,n_x_dach,w)
                else:
                    
                    x_train_batch   = x_train[:,index]
                    t_train_batch    = t_train[:,index]
                            
                    if u_train != None:
                        u_train_batch = u_train[:,index]
                        
                        
                    x_pred_batch, t_pred_batch, z_pred_batch, loss_train_t_s = step_predict(model=model,
                                     x_test=x_train_batch,
                                     t_test=t_train_batch,
                                     u_test= u_train_batch,
                                     index_t_s=index_t_s,
                                     num_pred=num_pred,
                                     method=method,
                                     options=options,
                                     step_size=step_size,
                                     time_obs_pred = time_obs_pred,
                                     loss_op=loss_op,
                                     n_x_dach=n_x_dach,
                                     w=w,
                                     data_step=data_step,
                                     training = True)    
                        
                        
                    
                    
                loss_train_list.append(loss_train_t_s.to(device))
            #####
            
            loss_train = torch.nanmean(torch.stack(loss_train_list))
            
            losses_train.append(loss_train.item())


            if not (torch.isinf(loss_train) or torch.isnan(loss_train)):

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                
        #######################################################################
        #validation
        
        model.eval()        
        with torch.no_grad():
            loss_val_tensor = torch.empty(len(list_index_t_s)).to(device)
            for t_s_i_val in range(len(list_index_t_s)):
                index_t_s = torch.tensor(list_index_t_s[t_s_i_val],device=device)
                
                if step_pred == False:
                
                    x_val_before, x_val_after        = x_val[:index_t_s+1], x_val[index_t_s:]
                    t_val_before,t_val_after         = t_val[:index_t_s+1], t_val[index_t_s:]
                                        
                    if u_val != None:    
                        u_val_before, u_val_after = u_val[:index_t_s+1], u_val[index_t_s:]
                    
                    x_pred_val, t_pred_val, z_pred_val, loss_val_t_s =  predict(model=model,
                            t_before=t_val_before,
                            x_before=x_val_before,
                            t_after=t_val_after,
                            x_after=x_val_after,
                            method=method,
                            options=options,
                            step_size=step_size,
                            u_before=u_val_before,
                            time_obs_pred = time_obs_pred,
                            index_t_s=index_t_s,
                            loss_op=loss_op,
                            n_x_dach=n_x_dach,
                            w=w,
                            data_step=data_step)
                    
                else: 
                    x_pred_val, t_pred_val, z_pred_val, loss_val_t_s = step_predict(model=model,
                                 x_test=x_val,
                                 t_test=t_val,
                                 u_test= u_val,
                                 index_t_s=index_t_s,
                                 num_pred=num_pred,
                                 method=method,
                                 options=options,
                                 step_size=step_size,
                                 time_obs_pred = time_obs_pred,
                                 loss_op=loss_op,
                                 n_x_dach=n_x_dach,
                                 w=w,
                                 data_step=data_step,
                                 training= False)

                loss_val_tensor[t_s_i_val] = loss_val_t_s.to(device)
                
            loss_val = torch.nanmean(loss_val_tensor)
            losses_val.append(loss_val.item())
            
            # Early stopping ######################################################
            stop_early, best_val_loss_early_stopping, early_stopping_counter = handle_early_stopping(
                loss_val.item(), best_val_loss_early_stopping, patience, early_stopping_counter, min_delta
            )
            if stop_early:
                print(f'Early stopping at epoch {epoch}')
                losses_val.append(loss_val.item())
                break   
    
            ## save results #######################################################   
            # save model with lowest test_loss
            if hyperopt == False:
                if loss_val.item() < best_val_loss:
                    best_val_loss = loss_val.item()
                    if save_results:
                        out_best = {'model':model,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': losses_train,
                              'val_loss': losses_val,
                              'epoch': epochs,
                              'batch_size':batch_size,
                              "learning_rate": learning_rate,
                              'z_pred_val':z_pred_val,
                              'x_pred_val':x_pred_val,
                              't_pred_val':t_pred_val,
                              'list_index_t_s':list_index_t_s}
                        torch.save(out_best,save_path+'_best')
            else:
                
                if loss_val.item() < best_val_loss:
                    best_val_loss = loss_val.item()
                    if save_results:
                        out_best = {'model':model,
                              'model_state_dict': model.state_dict(),
                              'list_index_t_s':list_index_t_s}
                        torch.save(out_best,save_path+'_best')
                
        ## save results ########################################################
        if hyperopt == False:        
            out={'model':model,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': losses_train,
                  'val_loss': losses_val,
                  'epoch': epochs,
                  'batch_size':batch_size,
                  "learning_rate": learning_rate,
                  'z_pred_val':z_pred_val,
                  'x_pred_val':x_pred_val,
                  't_pred_val':t_pred_val,
                  'list_index_t_s':list_index_t_s}
            if save_results:
                torch.save(out, save_path+'_all')
        else:
             out={'model':model,
                   'model_state_dict': model.state_dict(),
                   'list_index_t_s':list_index_t_s}
             if save_results:
                 torch.save(out, save_path+'_all')
    
    return out,losses_val
###############################################################################

def predict(model, t_before, x_before, t_after, x_after, method='euler',options=None, step_size=0.1,u_before=None,time_obs_pred = False,index_t_s=10,loss_op='default',n_x_dach=None,w=0,data_step=False):#, device
    device = x_before.device

    x_obs = x_before.transpose(0, 1)
        
    if u_before!= None:
        model.Node.u = u_before.transpose(0, 1)
    
    if time_obs_pred == True:
        x_obs = def_x_obs(x_obs=x_obs,time_obs=t_before.transpose(0, 1)).to(device)
            
    max_t = int(torch.max(t_after[~torch.isnan(t_after)]))
    t_pred = torch.arange(int(index_t_s), max_t+1, step=step_size, device=device)
    z_pred = model(t_pred, x_obs,method,options=options).to(device)

    if x_after.dim()>2:    
        x_pred = z_pred[...,:x_after.size(-1)]
    else:
           x_pred = z_pred[...,0]
           
    if data_step: x_after = data_time_step(x_after,step_size)   
    if data_step: t_after = data_time_step(t_after,step_size)
            
    if x_pred.dim()==2:
        loss = loss_function_1D(x_after.to(device), t_after.to(device), x_pred.to(device), t_pred.to(device),loss_op)
    else:
        loss = loss_function_sum(x_after.to(device), t_after.to(device), x_pred.to(device), t_pred.to(device),loss_op,n_x_dach,w)
    return x_pred.to(device), t_pred.to(device), z_pred.to(device), loss.to(device)

    
def step_predict(model, x_test, t_test,u_test, index_t_s,num_pred=2, method='euler',options=None, step_size=0.1,time_obs_pred = False,loss_op='default',n_x_dach=None,w=0,data_step=False,training=False):
    device = x_test.device
    x_pred_all = None
    loss_all = []
    for n in  range(int(x_test[index_t_s:].size(0)/(num_pred-1))):
        
        
        
        i_start = index_t_s+n*(num_pred-1)#+1
        i_end = i_start+ num_pred
        
        # t_test_before, t_test_after             = t_test[:i_start+1], t_test[i_start:i_end]
        # u_test_before, u_test_after             = u_test[:i_start+1], u_test[i_start:i_end]
        
        # if n == 0:
        #     x_test_before, x_test_after             = x_test[:i_start+1], x_test[i_start:i_end]
        # else:
        #     x_test_before, x_test_after             = torch.cat([x_test[:i_start+1] , x_pred_all]), x_test[i_start:i_end]
           
        t_test_before, t_test_after             = t_test[:i_start], t_test[i_start:i_end]
        u_test_before, u_test_after             = u_test[:i_start], u_test[i_start:i_end]
        
        if n == 0:
            x_test_before, x_test_after             = x_test[:i_start], x_test[i_start:i_end]
        else:
            x_test_before, x_test_after             = torch.cat([x_test[:i_start] , x_pred_all]), x_test[i_start:i_end]
               

        
        x_pred, t_pred, z_pred, loss = predict(model=model,
                                              step_size=step_size,
                                              time_obs_pred=time_obs_pred,
                                              t_before=t_test_before,
                                              x_before=x_test_before,
                                              t_after=t_test_after,
                                              x_after=x_test_after,
                                              method=method,
                                              u_before=u_test_before,
                                              index_t_s=i_start,
                                              n_x_dach=n_x_dach,
                                              w=w,data_step = data_step) 
        
        loss_all.append(loss)
        if n == 0:
            x_pred_all = x_pred.to(device)
            t_pred_all = t_pred.to(device)
            z_pred_all = z_pred.to(device)
    
        else:    
            x_pred_all = torch.cat([x_pred_all, x_pred[1:]]).to(device)
            t_pred_all = torch.cat([t_pred_all, t_pred[1:]]).to(device)
            z_pred_all = torch.cat([z_pred_all, z_pred[1:]]).to(device)
                
    if training == True:
        loss_out = torch.tensor(loss_all,requires_grad=True).mean().to(device)
    else:     
        loss_out = torch.tensor(loss_all,requires_grad=True).mean().to(device)
    return  x_pred_all.to(device), t_pred_all.to(device), z_pred_all.to(device), loss_out.to(device)



###############################################################################
### Plot ###
###############################################################################

def my_compute_prediction(x_test=None,t_test=None,u_test=None, index_t_s_i=None,model=None,n_x_dach=2, w=0,step_size=1,time_obs_pred=False,method='rk4',st=False,x_train=None,pred_func='new',
                          unscaled=False, step=None, variables_std=None,variables_mean=None,device = 'cpu',
                          offset =None, max_horizon = None,
                           model_decoder=None, test_X=None, test_toxic=None, test_treatments=None, covariables_x_test=None, test_time_covariates=None, active_entries_test=None,  a_loss=None,static=None,data_step=1.0,num_pred=2,step_pred=False):


    model.eval()   
    with torch.no_grad():
        # t_test_before, t_test_after     = t_test[:index_t_s_i+1].detach().clone(), t_test[index_t_s_i:].detach().clone()
        # x_test_before, x_test_after     = x_test[:index_t_s_i+1].detach().clone(), x_test[index_t_s_i:].detach().clone()
        # u_test_before, u_test_after     = u_test[:index_t_s_i+1].detach().clone(), u_test[index_t_s_i:].detach().clone()
        
        t_test_before, t_test_after     = t_test[:index_t_s_i].detach().clone(), t_test[index_t_s_i:].detach().clone()
        x_test_before, x_test_after     = x_test[:index_t_s_i].detach().clone(), x_test[index_t_s_i:].detach().clone()
        u_test_before, u_test_after     = u_test[:index_t_s_i].detach().clone(), u_test[index_t_s_i:].detach().clone()
        

        
        if step_pred == False:
                
            x_pred_i, _, _,_ = predict(model=model,
                                              step_size=step_size,
                                              time_obs_pred=time_obs_pred,
                                              t_before=t_test_before,
                                              x_before=x_test_before,
                                              t_after=t_test_after,
                                              x_after=x_test_after,
                                              method=method,
                                              u_before=u_test_before,
                                              index_t_s=index_t_s_i,
                                              n_x_dach=n_x_dach,
                                              w=w)#,
                                              #data_step=data_step)                

        else:
            x_pred_i, _,_,_ = step_predict(model=model,
                                                  x_test=x_test,
                                                  t_test=t_test,
                                                  u_test= u_test,
                                                  index_t_s=index_t_s_i,
                                                  num_pred=num_pred,
                                                  method=method,
                                                  step_size=step_size,
                                                  time_obs_pred = time_obs_pred,
                                                  n_x_dach=n_x_dach,
                                                  w=w,
                                                  data_step=data_step,
                                                  training = False)

                  
         
        X=x_test_after.transpose(1,0)
        pred_output_val = x_pred_i.transpose(1,0)             
         
         
    
    if st == True:
        for v_i in range(x_test.size(-1)):
            x_pred_i[..., v_i] = (x_pred_i[..., v_i]-np.nanmean(x_train[..., v_i]))/np.nanstd(x_train[..., v_i])
            x_test_after[..., v_i] = (x_test_after[..., v_i]-np.nanmean(x_train[..., v_i]))/np.nanstd(x_train[..., v_i])

    if unscaled:
        for v in range(pred_output_val.size(-1)):
            pred_output_val[:,:,v] = pred_output_val[:,:,v]*variables_std[v]+variables_mean[v]
        for v in range(X.size(-1)):
            X[:,:,v] = X[:,:,v]*variables_std[v]+variables_mean[v]    
            
    ### MSE, MAPE, WAPE, MAE ###
    mseloss = torch.nn.MSELoss()
    mse = torch.empty(size=pred_output_val.shape[1:],device=device)
    mse[:] = np.nan
    
    mapeloss=torchmetrics.MeanAbsolutePercentageError().to(device)
    mape = torch.empty(size=pred_output_val.shape[1:],device=device)
    mape[:]=np.nan
    
    wapeloss=torchmetrics.WeightedMeanAbsolutePercentageError().to(device)
    wape = torch.empty(size=pred_output_val.shape[1:],device=device)
    wape[:]=np.nan
    
    maeloss=torchmetrics.MeanAbsoluteError().to(device)
    mae = torch.empty(size=pred_output_val.shape[1:],device=device)
    mae[:]=np.nan
    
    if step is None:
        for i in range(mse.shape[0]):
            for j in range(mse.shape[1]):
                index_nan = (~X[:,i,j:j+1].isnan())
                
                mse[i,j]    = mseloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                mape[i,j]   = mapeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                wape[i,j]   = wapeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                mae[i,j]    = maeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                
    else:
        for i in range(mse.shape[0]):#step
            for j in range(mse.shape[1]):

                index_nan = (~X[:,i:i+step,j:j+1].isnan())    
                mse[i,j]    = mseloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                mape[i,j]   = mapeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                wape[i,j]   = wapeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                mae[i,j]    = maeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])

    
    ### RMSE ### 
    rmse=torch.sqrt(mse)
    
    X_std=X.clone()
    
    nrmse_sd = torch.empty(size=pred_output_val.shape[1:],device=device)
    nrmse_sd[:] = np.nan
    
    nrmse_mean = torch.empty(size=pred_output_val.shape[1:],device=device)
    nrmse_mean[:] = np.nan
    
    nrmse_iqr = torch.empty(size=pred_output_val.shape[1:],device=device)
    nrmse_iqr[:] = np.nan
    
    for j in range(X.shape[2]):
        X_std[:,:,j][(X[:,:,j].isnan())] = np.nan
    
    if step is None:

                
        dat_sd = torch.from_numpy(np.nanstd(X_std.cpu(),axis=0)).to(device)
        nrmse_sd = rmse/dat_sd[1:]
        
        dat_mean = torch.from_numpy(np.nanmean(X_std.cpu(),axis=0)).to(device)
        nrmse_mean = rmse/dat_mean[1:]
        
        dat_iqr = torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.75,axis=0)).to(device) - torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.25,axis=0)).to(device)
        nrmse_iqr=rmse/dat_iqr[1:]
    else:
        for i in range(mse.shape[0]):

            X_std_i = X_std[:,i:i+step,:]           
            dat_sd = torch.from_numpy(np.nanstd(X_std_i.cpu(),axis=(0,1))).to(device)
            dat_mean = torch.from_numpy(np.nanmean(X_std_i.cpu(),axis=(0,1))).to(device)
            dat_iqr = torch.from_numpy(np.nanquantile(X_std_i.cpu(),q=0.75,axis=(0,1))).to(device) - torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.25,axis=(0,1))).to(device)
       
            nrmse_sd[i] = rmse[i]/dat_sd
            nrmse_mean[i] = rmse[i]/dat_mean
            nrmse_iqr[i]=rmse[i]/dat_iqr

    return mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape



def heatmap_pred(x_test=None,t_test=None,u_test=None, list_index_t_s_pred=None,model=None,n_x_dach=2, w=0,step_size=1,time_obs_pred=False,method='rk4',st=False,x_train=False,pred_func = 'new',
                 index=0, offset=0, max_horizon=10,loss='rmse', rectilinear_index=None, step=None, unscaled=False, variables_std=None,variables_mean=None,variables=None,title=str(),save_link=None,save_map=None,load_map=None,vmin=None,vmax=None,colorbar=True,diagmultistep=False, dec_expand=False, med_dec=True, med_dec_start=False, invert=False,#):
                model_decoder=None, test_X=None, test_toxic=None, test_treatments=None, covariables_x_test=None, test_time_covariates=None, active_entries_test=None,  a_loss=None,static=None,dataset = None, val_seed =None, step_pred=False,data_step=1.0,heat_max=None):
    # Function to create a heatmap based on the prediction measures of the Decoder
    
    # Loading data (result data in diagonal form)
    if load_map is not None:
        #data2 = torch.load(load_map )
        with open(load_map, 'rb') as handle:
            heat_data = pickle.load(handle)
        #heat_data = heat_data[:,:]    
        if heat_max is not None:
            data = np.ma.masked_invalid(heat_data[:heat_max,:heat_max])#
        else:
            data = np.ma.masked_invalid(heat_data)#
        data2=ffill(data)    
            

    # Computing prediction measured
    else:
        heat_data = np.zeros(shape=[max_horizon,max_horizon])
        heat_data[:] = np.nan

        if val_seed !=None:
            heat_data = np.zeros(shape=[max_horizon,max_horizon])
            heat_data[:] = np.nan
            for j in range(val_seed):
                torch.manual_seed(j)
                np.random.seed(j)
                heat_data_all = []
                for i in range(max_horizon):

                    index_t_s_i = list_index_t_s_pred[i]
                    mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = my_compute_prediction(x_test=x_test,
                                                                                                        t_test=t_test,
                                                                                                        u_test=u_test,
                                                                                                        index_t_s_i=index_t_s_i,
                                                                                                        model=model,
                                                                                                        n_x_dach=n_x_dach,
                                                                                                        w=w,step_size=step_size,
                                                                                                        time_obs_pred=time_obs_pred,
                                                                                                        method=method,
                                                                                                        step = step,
                                                                                                        st=st,x_train=x_train,
                                                                                                        pred_func=pred_func,
                                                                                                        model_decoder=model_decoder,
                                                                                                        test_X=test_X,
                                                                                                        test_toxic=test_toxic, test_treatments=test_treatments,
                                                                                                        covariables_x_test=covariables_x_test,
                                                                                                        test_time_covariates=test_time_covariates,
                                                                                                        active_entries_test=active_entries_test,
                                                                                                        a_loss=a_loss,
                                                                                                        static=static,
                                                                                                        offset=offset,
                                                                                                        max_horizon=max_horizon,
                                                                                                        step_pred=step_pred,
                                                                                                        data_step=data_step)
                    if loss=='rmse':
                        heat_data[i,i+1:]=rmse[:-1,index].cpu().detach().numpy()
                    elif loss=='mse':
                        heat_data[i,i+1:]=mse[:-1,index].cpu().detach().numpy()
                    elif loss=='mae':
                        heat_data[i,i+1:]=mae[:-1,index].cpu().detach().numpy()
                    elif loss=='wape':
                        heat_data[i,i+1:]=wape[:-1,index].cpu().detach().numpy()
                    elif loss=='nrmse':
                        heat_data[i,i+1:]=nrmse_sd[:-1,index].cpu().detach().numpy()
                    heat_data_all.append(heat_data)    
                
            heat_data_std = np.nanstd(np.stack(heat_data_all),axis=0)   
            heat_data = np.nanmean(np.stack(heat_data_all),axis=0)       
            
            
        else:    
            for i in range(max_horizon):#-1

                index_t_s_i = list_index_t_s_pred[i]
                mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = my_compute_prediction(x_test=x_test,
                                                                                                    t_test=t_test,
                                                                                                    u_test=u_test,
                                                                                                    index_t_s_i=index_t_s_i,
                                                                                                    model=model,
                                                                                                    n_x_dach=n_x_dach,
                                                                                                    w=w,step_size=step_size,
                                                                                                    time_obs_pred=time_obs_pred,
                                                                                                    method=method,
                                                                                                    step = step,
                                                                                                    st=st,x_train=x_train,
                                                                                                    pred_func=pred_func,
                                                                                                    model_decoder=model_decoder,
                                                                                                    test_X=test_X,
                                                                                                    test_toxic=test_toxic, test_treatments=test_treatments,
                                                                                                    covariables_x_test=covariables_x_test,
                                                                                                    test_time_covariates=test_time_covariates,
                                                                                                    active_entries_test=active_entries_test,
                                                                                                    a_loss=a_loss,
                                                                                                    static=static,
                                                                                                    offset=offset,
                                                                                                    max_horizon=max_horizon,
                                                                                                    step_pred=step_pred,
                                                                                                    data_step=data_step)
                if loss=='rmse':
                    heat_data[i,i+1:]=rmse[:-1,index].cpu().detach().numpy()
                elif loss=='mse':
                    heat_data[i,i+1:]=mse[:-1,index].cpu().detach().numpy()
                elif loss=='mae':
                    heat_data[i,i+1:]=mae[:-1,index].cpu().detach().numpy()
                elif loss=='wape':
                    heat_data[i,i+1:]=wape[:-1,index].cpu().detach().numpy()
                elif loss=='nrmse':
                    heat_data[i,i+1:]=nrmse_sd[:-1,index].cpu().detach().numpy()

        #heat_data = heat_data[1:,1:]
        data = np.ma.masked_invalid(heat_data)#[1:,1:]
        data2=ffill(data)
    
    if not invert:
        data2 = np.transpose(data2)
    
    # Creating plots
    if colorbar:
        fig, ax = plt.subplots(figsize=(5,4),dpi=400)
    else:
        fig, ax = plt.subplots(figsize=(4,4),dpi=400)
    
    if vmin is None:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=0, vmax=np.nanmax(data))
    else:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=vmin, vmax=vmax)
    
    if colorbar:
        fig.colorbar(heatmap)
    
    # Setting some plotting options
    ax.set_xticks(np.arange(data2.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data2.shape[0])+0.5, minor=False)
    
    if max_horizon>72:
        a=np.arange(0,data2.shape[0],10)
        b=range(0,data2.shape[1],10)
    else:
        a=np.arange(0,data2.shape[0]+1,5)
        b=range(0,data2.shape[1]+1,5)
        
    ax.xaxis.set(ticks=a, ticklabels=a)

    ax.yaxis.set(ticks=b,ticklabels=b)
    
    if invert:
        if dataset == 'DoseAI':ax.set_ylabel('Observed until treatment cycle y in months',fontsize= 13)
        if dataset == 'mimic4':ax.set_ylabel('Observed until hour y',fontsize= 13)
        
        if dataset =='DoseAI':ax.set_xlabel('Forecast horizon in treatment cycles',fontsize= 13)
        if dataset == 'mimic4':ax.set_xlabel('Forecast horizon in hours',fontsize= 13)
    
    else:
        if dataset == 'DoseAI':ax.set_xlabel('Observed until treatment cycle x in months',fontsize= 13)
        if dataset == 'mimic4':ax.set_xlabel('Observed until hour x',fontsize= 13)
        
        if dataset == 'DoseAI':ax.set_ylabel('Forecast horizon in treatment cycles',fontsize= 13)
        if dataset == 'mimic4':ax.set_ylabel('Forecast horizon in hours',fontsize= 13)
    
    
    if loss=='rmse':
        ax.set_title(title+'Root Mean Square Error')
    elif loss=='mse':
        ax.set_title(title+'Mean Square Error',fontsize= 13)
    elif loss=='mae':
        ax.set_title(title+'Mean Absolute Error')
    elif loss=='wape':
        ax.set_title(title+'WAPE')
    elif loss=='nrmse':
        ax.set_title(title+'NRMSE')
    
    if save_link is not None:
        plt.savefig(save_link)
    if save_map is not None:
        with open(save_map, 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if val_seed !=None:
        return (heat_data, heat_data_std)
    else:
        return heat_data
     
    
def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out
