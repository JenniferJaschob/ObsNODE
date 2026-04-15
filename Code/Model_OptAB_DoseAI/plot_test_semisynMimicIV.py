import torch
import numpy as np
import matplotlib.pyplot as plt
#import utils_paper_MimicIV as utils_paper
import pandas as pd
import pickle            
    
import torch
import numpy as np
# import utils_paper_new as utils_paper
# from  utils_paper_new import * 


import utils_paper_MimicIV as utils_paper
from  utils_paper_MimicIV import * 

import pandas as pd
import pickle

######
print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

path_data = '/Users/jaschob/Desktop/semi_syn_mimic4/'

save2 = '/Users/jaschob/Desktop/out2/semi_syn_optab/res_semi_syn_mimic4/Trained_Decoder_seed_'+str(num_seed_i)+'_semisynmimic4_15.pth'
save = '/Users/jaschob/Desktop/save_pkl/res_semi_syn_mimic4/Trained_Encoder_seed_'+str(num_seed_i)+'_semisynmimic4.pth'
# save2 = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Decoder_seed_'+str(num_seed_i)+'_MimicIV.pth'
# save = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Encoder_seed_'+str(num_seed_i)+'_MimicIV.pth'

folder_h= '/Users/jaschob/Desktop/mimic4_heat/OptAB_semisynMimic4/'

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
stop
# # hyperparameters of the Encoder trail 19
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


# # Hyperparameters of Decoder trail 15
hidden_channels_dec = 27
batch_size_dec = 1000
hidden_states_dec = 974
lr_dec = 0.002392251517863519
activation_dec = 'tanh'
num_depth_dec = 20
pred_act_dec = 'leakyrelu'
pred_states_dec = 203
pred_depth_dec = 2

#Hyperparameters of Decoder trail 1
# hidden_channels_dec = 25
# batch_size_dec = 1000
# hidden_states_dec = 825
# lr_dec = 0.0006940775886326423
# activation_dec = 'identity'
# num_depth_dec = 9
# pred_act_dec = 'leakyrelu'
# pred_states_dec = 403
# pred_depth_dec = 2

offset=0
rectilinear_index=0


data_thresh = torch.zeros(1)
model = utils_paper.NeuralCDE(input_channels=data_covariables_train.size(-1),
                              hidden_channels=hidden_channels,
                              hidden_states=hidden_states,
                              output_channels=1,
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

# # Initializing and loading the Encoder
model.load_state_dict(torch.load(save, map_location=torch.device("cpu")))
model=model.to(device)


#########


z0_hidden_dimension_dec = hidden_channels+5

model_decoder = utils_paper.NeuralCDE(input_channels=1,
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


model_decoder.load_state_dict(torch.load(save2, map_location=torch.device("cpu")))

model_decoder=model_decoder.to(device)


save_res = False
save_heat = False




unscaled=False
offset=0
max_horizon=23
invert = True
colorbar = True

#########
list_num_seed = [0,1,2,3,4]#[0,1,3,5]#[0,1,2,3,4] # 4 und 2 noch nicht fertig
res_dic0_all, res_dic1_all = [], []
for seed_i in range(len(list_num_seed)):        
    num_seed_i = list_num_seed[seed_i]
    
    torch.manual_seed(num_seed_i)
    np.random.seed(num_seed_i)
    
    #save2 = '/Users/jaschob/Desktop/save_pkl/res_semi_syn_mimic4/Trained_Decoder_seed_'+str(num_seed_i)+'_semisynmimic4_15.pth'
    #save = '/Users/jaschob/Desktop/save_pkl/res_semi_syn_mimic4/Trained_Encoder_seed_'+str(num_seed_i)+'_semisynmimic4.pth'
    save = '/Users/jaschob/Desktop/train/Trained_Encoder_seed_'+str(num_seed_i)+'_semisynmimic4.pth'
    save2 = '/Users/jaschob/Desktop/train/Trained_Decoder_seed_'+str(num_seed_i)+'_semisynmimic4_15.pth'
    
    # save2 = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Decoder_seed_'+str(num_seed_i)+'_MimicIV.pth'
    # save = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Encoder_seed_'+str(num_seed_i)+'_MimicIV.pth'

    model.load_state_dict(torch.load(save, map_location=torch.device("cpu")))
    model_decoder.load_state_dict(torch.load(save2, map_location=torch.device("cpu")))


    res_dic0 = res_dic0= utils_paper.heatmap_pred_dec(model,
                                           model_decoder,
                                           offset=offset,
                                           max_horizon=max_horizon,
                                           loss='rmse',
                                           unscaled=unscaled,
                                           validation_output=data_X_test,
                                           validation_toxic=data_toxic_test,
                                           validation_treatments=data_treatment_test,
                                           covariables=data_covariables_test,
                                           time_covariates=data_time_test,
                                           active_entries=data_active_test,
                                           rectilinear_index=rectilinear_index,
                                           step=None,
                                           dec_expand=True,
                                           sofa_expand=True,
                                           med_dec=False,
                                           med_dec_start=True,
                                           save_link=None,
                                           load_map=None,
                                           title=' Value 0',
                                           index=0,
                                           colorbar=colorbar,
                                           invert=invert)
    
    if save_heat: plt.savefig(save_res0 +'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight') 
    if save_res:
        with open(save_res0 +'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
            pickle.dump(res_dic0, f)
            
            
    res_dic0_all.append(res_dic0) 
    

save_res0 = folder_h+'heatmap_pred__val0__horizon_'+str(max_horizon)#'/Users/jaschob/Desktop/mimic4_heat/OptAB_seed/'

heat_data0_std = np.nanstd(np.stack(res_dic0_all),axis=0)   
heat_data0_mean = np.nanmean(np.stack(res_dic0_all),axis=0)      
with open(save_res0 +'_seed_all.pkl', 'wb') as f:
    pickle.dump(res_dic1_all, f)
with open(save_res0 +'_mean.pkl', 'wb') as f:
    pickle.dump(heat_data0_mean, f)    
with open(save_res0 +'_std.pkl', 'wb') as f:
    pickle.dump(heat_data0_std, f)    


####
save_heat = True

res_dic0_mean = utils_paper.heatmap_pred_dec(model,
                                             model_decoder,
                                             validation_output=data_X_test,
                                             validation_toxic=None,
                                             validation_treatments=data_treatment_test,
                                             covariables=data_covariables_test,
                                             time_covariates=data_time_test,
                                             active_entries=data_active_test,
                                             static=None,
                                             title ='Value 0',
                                             load_map=save_res0 +'_mean.pkl',
                                             invert=invert
                                             ,colorbar=colorbar)
if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  

res_dic0_std = res_dic0_mean = utils_paper.heatmap_pred_dec(model,
                                             model_decoder,
                                             validation_output=data_X_test,
                                             validation_toxic=None,
                                             validation_treatments=data_treatment_test,
                                             covariables=data_covariables_test,
                                             time_covariates=data_time_test,
                                             active_entries=data_active_test,
                                             static=None,
                                             title ='Value 0 ',
                                             load_map=save_res0 +'_std.pkl',
                                             invert=invert,
                                             colorbar=colorbar)
if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  


#####

vmin=0
vmax=1.0
save_heat=True
heat_max = 12#None
res_dic0_mean = utils_paper.heatmap_pred_dec(model,
                                             model_decoder,
                                             validation_output=data_X_test,
                                             validation_toxic=None,
                                             validation_treatments=data_treatment_test,
                                             covariables=data_covariables_test,
                                             time_covariates=data_time_test,
                                             active_entries=data_active_test,
                                             static=None,
                                             title =' Value 0 ',
                                             load_map=save_res0 +'_mean.pkl',
                                             invert=invert
                                             ,colorbar=colorbar,
                                             vmax=vmax,vmin=vmin,heat_max=heat_max)
#if save_heat: plt.savefig(save_res0 +'_mean_'+str(vmax)+'.png', bbox_inches='tight') 
if save_heat: plt.savefig(save_res0 +'_mean_'+str(vmax)+'_heat_max_'+str(heat_max)+'.png', bbox_inches='tight')  

#print
for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,3)),np.flipud(np.round(res_dic0_std.T,3))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.3f}'
            s_str = f'{s:.3f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    


for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,2)),np.flipud(np.round(res_dic0_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    
