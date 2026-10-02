#%%
import torch
import numpy as np
import optuna
import joblib
import pickle

import matplotlib.pyplot as plt


import os
#os.chdir('/home/jaschob/server/NODE_ObsNODE/')
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Model_observableNODE_paper')
from utils_NODE import *

num_seed_i = 105
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save= 'my_model_semisynmimicIV_NODE.pkl'

#path_data = '/home/jaschob/server/semi_syn_mimic4/'
path_data = '/Users/jaschob/Desktop/Paper1/semi_syn_mimic4/'
with open(path_data + "x_validate_syn_treat.pkl", "rb") as datei:
    x_validate = pickle.load(datei)        
with open(path_data + "x_train_syn_treat.pkl", "rb") as datei:
    x_train = pickle.load(datei)    
with open(path_data + "x_test_syn_treat.pkl", "rb") as datei:
    x_test = pickle.load(datei)    
          
with open(path_data + "u_validate_syn.pkl", "rb") as datei:
    u_validate = pickle.load(datei)
with open(path_data + "u_train_syn.pkl", "rb") as datei:
    u_train = pickle.load(datei)
with open(path_data + "u_test_syn.pkl", "rb") as datei:
    u_test = pickle.load(datei)    
        
with open(path_data + "t_validate.pkl", "rb") as datei:
    t_validate = pickle.load(datei)
with open(path_data + "t_train.pkl", "rb") as datei:
    t_train = pickle.load(datei)
with open(path_data + "t_test.pkl", "rb") as datei:
    t_test = pickle.load(datei)

num_time = 48
t_validate, x_validate =    t_validate[:num_time].to(torch.float).to(device), x_validate[:num_time].to(torch.float).to(device)
t_train, x_train =          t_train[:num_time].to(torch.float).to(device), x_train[:num_time].to(torch.float).to(device)
t_test, x_test =            t_test[:num_time].to(torch.float).to(device), x_test[:num_time].to(torch.float).to(device)
u_train, u_test, u_validate =  u_train[:num_time].to(torch.float).to(device),  u_test[:num_time].to(torch.float).to(device), u_validate[:num_time].to(torch.float).to(device)


params_dic = {'n_z': 2,
                'batch_size': 100,
                'learning_rate': 0.001,
                'hidden_dim_obs': 64,
                'hidden_dim_node': 64,
                'num_layers_node': 3,
                'activation_node': 'leakyrelu'}
#n_x = x_test.size(2)
#n_u = u_train.size(-1)
#n_t = 0
n_z, n_x, n_u, n_t = params_dic['n_z'],2,2,0
    
n_x_dach = n_x
w=0
    
batch_size = params_dic['batch_size']
learning_rate = params_dic['learning_rate']
patience = 30
epochs = 50

method = 'rk4'
options=None

optimizer_func = 'Adam'
loss_op= 'default'

step_size=1.0
data_step = True      
adjoint_method='original'

hidden_dim_obs=params_dic['hidden_dim_obs']
dropout = 0.0 

hidden_sizes_node = params_dic['hidden_dim_node']
num_layers_node = params_dic['num_layers_node']
activation_node = params_dic['activation_node']

time_obs_pred = False
list_index_t_s = [12,24,36]

# define Model ################################################################
myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
      
myNODE = MyNeuralODE_withNNs( activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
model = MyModel_adjoint(myNODE, myObserver).to(device)

save_out = True
out,loss = train_diff_ts(model_node=myNODE,
                                    model_observer=myObserver,
                                    u_train=u_train,u_val=u_validate,
                                    time_obs_pred=time_obs_pred,
                                    x_train=x_train,x_val=x_validate,
                                    t_train=t_train,t_val=t_validate,
                                    method=method,
                                    options= options,
                                    patience=patience,
                                    optimizer_func=optimizer_func,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    batch_size = batch_size,
                                    list_index_t_s=list_index_t_s,
                                    save_path=save,
                                    step_size=step_size,
                                    loss_op=loss_op,
                                    hyperopt=False,
                                    n_x_dach=n_x_dach,
                                    w=w,
                                    data_step = data_step)
    
#%%
save_out = True 
save_path2 = '/Users/jaschob/Desktop/Out_NODE/model_state_dict_seed_'+str(num_seed_i)+'_semisynmimicIV.pth'
if save_out == True: torch.save(model.state_dict(), save_path2)
    
save_path3 = '/Users/jaschob/Desktop/Out_NODE/out_seed_'+str(num_seed_i)+'_semisynmimicIV.pth'
if save_out == True: torch.save(out, save_path3)

# %%
