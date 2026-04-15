import torch
import numpy as np
import utils_paper_MimicIV as utils_paper

import pandas as pd
import pickle

######
print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

path_data = '/home/jaschob/server/semi_syn_mimic4/'
save_path = '/home/jaschob/server/OptAB/res_semi_syn_mimic4/'

save = save_path + 'Trained_Encoder_seed_1_semisynmimic4.pth'
save2 = save_path +'Trained_Decoder_seed_'+str(num_seed_i)+'_semisynmimic4.pth'

#data
    
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

data_time_train = t_train[:-1,:,0:1].transpose(0,1).to(device)
data_time_test = t_validate[:-1,:,0:1].transpose(0,1).to(device)    

# # hyperparameters of the Encoder
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

# Hyperparameters of Decoder 
hidden_channels_dec = 25
batch_size_dec = 1000
hidden_states_dec = 825
lr_dec = 0.0006940775886326423
activation_dec = 'identity'
num_depth_dec = 9
pred_act_dec = 'leakyrelu'
pred_states_dec = 403
pred_depth_dec = 2

offset=0
rectilinear_index=0


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
                              device=device)#,


# # Initializing and loading the Encoder
model.load_state_dict(torch.load(save))
model=model.to(device)


#########

z0_hidden_dimension_dec = hidden_channels+5

model_decoder = utils_paper.NeuralCDE(input_channels=1,#data_X_train.size(-1),
                                      hidden_channels=hidden_channels_dec,
                                      hidden_states=hidden_states_dec,
                                      output_channels=1,
                                      z0_dimension_dec=z0_hidden_dimension_dec,
                                      activation=activation_dec,
                                      num_depth=num_depth_dec,
                                      pos=True,
                                      thresh=data_thresh,
                                      pred_comp=True,
                                      pred_act=pred_act_dec,
                                      pred_states=pred_states_dec,
                                      pred_depth=pred_depth_dec,
                                      treatment_options=data_treatment_train.size(-1),
                                      device = device)#

model_decoder=model_decoder.to(device)

####

max_horizon=121
b=list(range(max_horizon))
offset_train_list=[]
for i in range(0,len(b),10):
    if i<=20:
        c=np.random.choice(b[i:i+10],size=2,replace=False)
        offset_train_list.append(c[0])
        offset_train_list.append(c[1])
    else:
        offset_train_list.append(np.random.choice(b[i:i+10],replace=False))
offset_train_list.sort()


loss = utils_paper.train_dec_offset(model,
                                    model_decoder,
                                    offset=offset,
                                    max_horizon=max_horizon,
                                    lr=lr_dec,
                                    batch_size=batch_size_dec,
                                    patience=10,
                                    delta=0.0001,
                                    max_epochs=1000,
                                    weight_loss=True,
                                    train_output=data_X_train,
                                    train_toxic=None,#data_toxic_train,
                                    train_treatments=data_treatment_train,
                                    covariables=data_covariables_train,
                                    time_covariates=data_time_train,
                                    active_entries=data_active_train,
                                    static=None,
                                    
                                    rectilinear_index=rectilinear_index,
                                    validation_output=data_X_test,
                                    validation_toxic=None,#data_toxic_test,
                                    validation_treatments=data_treatment_test,
                                    covariables_val=data_covariables_test,
                                    validation_time_covariates=data_time_test,
                                    active_entries_val=data_active_test,
                                    static_val=None,
                                    
                                    offset_train_list=offset_train_list,
                                    offset_val_list=None,
                                    early_stop_path=save2,
                                    dec_expand=True,
                                    sofa_expand=True,
                                    med_dec=False,
                                    med_dec_start=True,
                                    min_epoch=50)


