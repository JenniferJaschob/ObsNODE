import torch
import numpy as np
import optuna
import joblib
import pickle

from utils_observableNODE import *


num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


save =  'my_model_MimicIV.pkl'

path_data = '/Users/jaschob/Desktop/server_26525/server/data_mimic/'
with open(path_data + "data_val_mimiciv2.pkl", "rb") as datei:
    data_validate = pickle.load(datei)
        
with open(path_data + "data_train_mimiciv2.pkl", "rb") as datei:
    data_train = pickle.load(datei)
    
with open(path_data + "data_test_mimiciv2.pkl", "rb") as datei:
    data_test = pickle.load(datei)
    
# Steuerung u    
with open(path_data + "data_val_mimiciv_value_u2.pkl", "rb") as datei:
    u_validate = pickle.load(datei)
        
with open(path_data + "data_train_mimiciv_value_u2.pkl", "rb") as datei:
    u_train = pickle.load(datei)
    
with open(path_data + "data_test_mimiciv_value_u2.pkl", "rb") as datei:
    u_test = pickle.load(datei)    
        
        
t_validate, x_validate, _  =  data_validate
t_train, x_train, _        =  data_train
t_test, x_test, _          =  data_test
    
val_list= [5,9,14,15]+[4,6,7,8,10,11,12,13]+list(range(19,30))+list(range(30,38))+[38,39,40]
num_time = 48
    
t_validate, x_validate =    t_validate[:num_time,:,val_list].to(torch.float).to(device), x_validate[:num_time,:,val_list].to(torch.float).to(device)
t_train, x_train =          t_train[:num_time,:,val_list].to(torch.float).to(device), x_train[:num_time,:,val_list].to(torch.float).to(device)
t_test, x_test =            t_test[:num_time,:,val_list].to(torch.float).to(device), x_test[:num_time,:,val_list].to(torch.float).to(device)
    
u_train[torch.isnan(u_train)],u_test[torch.isnan(u_test)],u_validate[torch.isnan(u_validate)] = 0,0,0
u_validate, u_test, u_train = u_validate[:num_time,:,:].to(torch.float).to(device), u_test[:num_time,:,:].to(torch.float).to(device), u_train[:num_time,:,:].to(torch.float).to(device)

params_dic = {'n_z': 34,
  'batch_size': 500,
  'learning_rate': 0.001,
  'hidden_dim_obs': 256,
  'hidden_dim_node': 128,
  'num_layers_node': 8,
  'activation_node': 'leakyrelu'}

n_z = x_test.size(2)
n_x = x_test.size(2)
n_u = u_train.size(-1)
n_t = 0
    
n_x_dach = 4
w=0
    
batch_size = params_dic['batch_size']
learning_rate = params_dic['learning_rate']
patience = 30
epochs = 50

method = 'rk4'
options=None

optimizer_func = 'Adam'
loss_op= 'default'

# num=15
step_size=1.0
data_step = True      
adjoint_method='original'

hidden_dim_obs=params_dic['hidden_dim_obs']
dropout = 0.0 

hidden_sizes_node = params_dic['hidden_dim_node']
num_layers_node = params_dic['num_layers_node']
activation_node = params_dic['activation_node']

hidden_sizes_helperNN = hidden_sizes_node 
num_layers_helperNN = num_layers_node
activation_helperNN = activation_node

time_obs_pred = False
if num_time == 48:
    list_index_t_s= [4,6,8,10] 
elif num_time==12:
    list_index_t_s = [4,6,8,10]    
else:   
    list_index_t_s = [4,6,8,10]

# define Model ################################################################
myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
    
helperNN_list = [] 
for j in range(1,n_z):
    helperNN= MyhelperNN(activation = activation_helperNN ,hidden_sizes =hidden_sizes_helperNN ,num_layers=num_layers_helperNN,n_u=n_u,n_z_i=j*n_x,output_size=n_x).to(device)
    helperNN_list.append(helperNN)
        
myNODE = MyObservableNeuralODE_withNNs(helperNN_list=helperNN_list, activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
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
                                    data_step = data_step,step_pred=True,num_pred=2)
    
    
    
save_path2 = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_Mimic4/model_state_dict_seed_'+str(num_seed_i)+'_MimicIV.pth'
if save_out == True: torch.save(model.state_dict(), save_path2)


save_path3 = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_Mimic4/out_seed_'+str(num_seed_i)+'_MimicIV.pth'
if save_out == True: torch.save(out, save_path3)