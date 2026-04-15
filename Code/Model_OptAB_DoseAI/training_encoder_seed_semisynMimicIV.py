import torch
import numpy as np
import utils_paper_MimicIV as utils_paper

import pandas as pd
import pickle


print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_seed_i = 0
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

save = str('Trained_Encoder_seed_'+str(num_seed_i)+'_semisynmimic4.pth')

# data
path_data = '/home/jaschob/server/semi_syn_mimic4/'

    
with open(path_data + "x_validate_syn.pkl", "rb") as datei:
    x_validate = pickle.load(datei)        
with open(path_data + "x_train_syn.pkl", "rb") as datei:
    x_train = pickle.load(datei)    
with open(path_data + "x_test_syn.pkl", "rb") as datei:
    x_test = pickle.load(datei)    
    
# Steuerung u    
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

data_X_train = x_train[1:].transpose(0,1).to(device)
data_covariables_train = torch.cat([x_train,u_train,t_train[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)
data_treatment_train = u_train.transpose(0,1).to(device)
data_active_train =   ~x_train.transpose(0,1).isnan().to(device)
data_toxic_train = None

data_X_test = x_validate[1:].transpose(0,1).to(device)
data_covariables_test = torch.cat([x_validate,u_validate,t_validate[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)
data_treatment_test = u_validate.transpose(0,1).to(device)
data_active_test =  ~x_validate.transpose(0,1).isnan().to(device)
data_toxic_test = None


hidden_channels = 7
batch_size = 1000
hidden_states = 174
lr = 0.0015972840572993194
activation = 'leakyrelu'
num_depth = 14
pred_act = 'leakyrelu'
pred_states = 275
pred_depth = 4

pred_comp=True



data_thresh = torch.zeros(1)
model = utils_paper.NeuralCDE(input_channels=data_covariables_train.size(-1),
                              hidden_channels=hidden_channels,
                              hidden_states=hidden_states,
                              output_channels=1,#1,
                              treatment_options=data_treatment_train.size(-1),
                              activation = activation,
                              num_depth=num_depth,
                              interpolation="linear",
                              pos=True,
                              thresh=data_thresh,
                              pred_comp=pred_comp,
                              pred_act=pred_act,
                              pred_states=pred_states,
                              pred_depth=pred_depth,
                              device=device)
model=model.to(device)


loss = utils_paper.train(model,
                         weight_loss=True,
                         lr=lr,
                         batch_size=batch_size,
                         patience=10,
                         delta=0.0001,
                         max_epochs=1000,
                         train_output=data_X_train,
                         train_toxic=data_toxic_train,
                         train_treatments=data_treatment_train,
                         covariables=data_covariables_train,
                         active_entries=data_active_train,
                         
                         validation_output=data_X_test,
                         validation_toxic=data_toxic_test,
                         validation_treatments=data_treatment_test,
                         covariables_val=data_covariables_test,
                         active_entries_val=data_active_test,
                         static=None,
                         static_val=None,
                         rectilinear_index=0,
                         early_stop_path=save)
