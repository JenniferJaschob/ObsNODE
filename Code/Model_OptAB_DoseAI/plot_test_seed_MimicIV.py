import torch
import numpy as np
import matplotlib.pyplot as plt
import utils_paper_MimicIV as utils_paper
import pandas as pd
import pickle            


print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_seed_i = 4
print(num_seed_i)
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)
    
folder_data = '/Users/jaschob/Desktop/data_m4/data_preproc_phil/data_prep_three/'
#folder_h = '/Users/jaschob/Desktop/mimic4_heat/OptAB_mimic4_seed/MimicIV_new_24/'
folder_h = '/Users/jaschob/Desktop/mimic4_heat/OptAB_mimic4_seed/MimicIV_new_12/'

save = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Encoder_seed_'+str(num_seed_i)+'_MimicIV.pth'#'/Volumes/T7/save_data_mac/serverOptAB/save_pth/Trained_Encoder_seed'+str(num_seed_i)+'.pth'
save2 = '/Users/jaschob/Desktop/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_MimicIV/Trained_Decoder_seed_'+str(num_seed_i)+'_MimicIV.pth'#'/Volumes/T7/save_data_mac/serverOptAB/save_pth/Trained_Decoder_seed'+str(num_seed_i)+'.pth'

with open(folder_data+'sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)
# Tensor of patient data, size: number of patients x timepoints x variables (including one dimension corresponding to time)

with open(folder_data+'sepsis_all_1_keys.pkl', 'rb') as handle:
    key_times = pickle.load(handle)
# Dictionary where keys correspond to the index and the values correspond to the time in hours

with open(folder_data+'sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)
# List of all variable names of the time-dependent variables

with open(folder_data+'sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)
# Tensor of static variables, size: number of patients x static_variables

with open(folder_data+'sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)
# List of all static variable names

with open(folder_data+'sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)
# list of means of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)

with open(folder_data+'sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
# list of standard deviations of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)
    
with open(folder_data+'sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)
    
with open(folder_data+'sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)
# list of index variables for the variables to be standardized

# hyperparameters of the Encoder
hidden_channels = 17
#batch_size = 500
hidden_states = 33
#lr = 0.0050688746606452565
activation = 'tanh'
num_depth = 15
pred_act = 'tanh'
pred_states = 128
pred_depth = 1
pred_comp=True


# Threshold for the model to compute only positive outputs (via softplus)
data_thresh = ((0-variables_mean)/variables_std)[[variables.index("SOFA"),variables.index("creatinine"),variables.index("bilirubin_total"),variables.index("alt")]]

# Initializing and loading the Encoder
model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables),device=device)
model=model.to(device)
model.load_state_dict(torch.load(save,map_location=torch.device('cpu')))

# Hyperparameters of Decoder
hidden_channels_dec = 17
batch_size_dec = 1000
hidden_states_dec = 33
lr_dec = 0.0050688746606452565
activation_dec = 'tanh'
num_depth_dec = 15
pred_act_dec = 'tanh'
pred_states_dec = 128
pred_depth_dec = 1

offset=0

# determined by data
output_channels=4 
input_channels_dec=1 #
z0_hidden_dimension_dec = hidden_channels + 1 + static_tensor.shape[-1] + 6

# Initializing Decoder
model_decoder = utils_paper.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=z0_hidden_dimension_dec,activation=activation_dec,num_depth=num_depth_dec, pos=True, thresh=data_thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec, treatment_options=3, device = device)
model_decoder=model_decoder.to(model_decoder.device)
model_decoder.load_state_dict(torch.load(save2,map_location=torch.device('cpu')))

rectilinear_index=0

#### "Pre-Processing of data"

## "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_test])
key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times_train={list(key_times.keys())[x]: key_times[x] for x in key_times_index}
key_times=key_times_train

#Extracting outcome and side-effects from data and setting device
data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test].to(model.device)
data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic=data_toxic[:,:,None][indices_test].to(model.device)
data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2=data_toxic2[:,:,None][indices_test].to(model.device)
data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3=data_toxic3[:,:,None][indices_test].to(model.device)
data_toxic_test = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)

#Extracting treatments and side-effects from data and setting device
data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment=data_treatment[:,:,None][indices_test].to(model.device)
data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2=data_treatment2[:,:,None][indices_test].to(model.device)
data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3=data_treatment3[:,:,None][indices_test].to(model.device)
data_treatment_test = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)

#Extracting the covariables
data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test].to(model.device)

#Normalizing the missing masks to one
time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max

# Selection of training and test data
data_time_test = data[:,:list(key_times.keys())[-1],0:1][indices_test].to(model.device)
data_active_test = ~data[:,list(key_times.values()),1:2].isnan()[indices_test].to(model.device)
data_static_test=static_tensor[indices_test].to(model.device,dtype=torch.float32)    


# Compute unscaled data
data_toxic_test_unscaled=data_toxic_test.clone()
data_toxic_test_unscaled[:,:,0] = data_toxic_test_unscaled[:,:,0]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
data_toxic_test_unscaled[:,:,1] = data_toxic_test_unscaled[:,:,1]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
data_toxic_test_unscaled[:,:,2] = data_toxic_test_unscaled[:,:,2]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]



unscaled=False

save_fig = True
save_res = True


offset=0
max_horizon=12
step=None
save_link = None
load_map = None
vmin=0
vmax=None#0.7#1
invert = True
colorbar=True#False
stop
title='Sofa-Score '
res_dic0= utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='rmse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=0,colorbar=colorbar, invert=invert)
if save_fig: plt.savefig(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse.png', bbox_inches='tight')    
if save_res:
    with open(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse_data.pkl', 'wb') as f:
        pickle.dump(res_dic0, f)

step=None
title='Creatinine '
res_dic1=utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='rmse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=1,colorbar=colorbar, invert=invert)
if save_fig: plt.savefig(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse.png', bbox_inches='tight')    
if save_res:
    with open(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse_data.pkl', 'wb') as f:
        pickle.dump(res_dic1, f)
        
step=None
title='Bilirubin-total '
res_dic2=utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='rmse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=2,colorbar=colorbar, invert=invert)
if save_fig: plt.savefig(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse.png', bbox_inches='tight')    
if save_res:
    with open(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse_data.pkl', 'wb') as f:
        pickle.dump(res_dic2, f)

step=None
title='ALT '
res_dic3=utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='rmse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=3,colorbar=colorbar, invert=invert)
if save_fig: plt.savefig(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse.png', bbox_inches='tight')    
if save_res:
    with open(folder_h+'heatmap_pred_phil_'+title+'_horizon'+str(max_horizon)+'_step'+str(step)+'_seed_'+str(num_seed_i)+'_rmse_data.pkl', 'wb') as f:
        pickle.dump(res_dic3, f)

#########

stop
res_dic0_all, res_dic1_all,res_dic2_all, res_dic3_all = [], [], [], []
seed_i = [0,1,2,3,4]
for i in range(len(seed_i)):
    seed = seed_i[i]
    with open(folder_h+'heatmap_pred_phil_Sofa-Score _horizon'+str(max_horizon)+'_stepNone_seed_'+str(seed)+'_rmse_data.pkl', 'rb') as f:
        res_dic0 = pickle.load(f)
        
    with open(folder_h+'heatmap_pred_phil_Creatinine _horizon'+str(max_horizon)+'_stepNone_seed_'+str(seed)+'_rmse_data.pkl', 'rb') as f:
        res_dic1 = pickle.load(f)

    with open(folder_h+'heatmap_pred_phil_Bilirubin-total _horizon'+str(max_horizon)+'_stepNone_seed_'+str(seed)+'_rmse_data.pkl', 'rb') as f:
        res_dic2 = pickle.load(f)

    with open(folder_h+'heatmap_pred_phil_ALT _horizon'+str(max_horizon)+'_stepNone_seed_'+str(seed)+'_rmse_data.pkl', 'rb') as f:
        res_dic3 = pickle.load(f)        
    
    res_dic0_all.append(res_dic0)
    res_dic1_all.append(res_dic1)
    res_dic2_all.append(res_dic2)
    res_dic3_all.append(res_dic3)
    
heat_data0_std = np.nanstd(np.stack(res_dic0_all),axis=0)   
heat_data0_mean = np.nanmean(np.stack(res_dic0_all),axis=0)      

heat_data1_std = np.nanstd(np.stack(res_dic1_all),axis=0)   
heat_data1_mean = np.nanmean(np.stack(res_dic1_all),axis=0)      

heat_data2_std = np.nanstd(np.stack(res_dic2_all),axis=0)   
heat_data2_mean = np.nanmean(np.stack(res_dic2_all),axis=0)

heat_data3_std = np.nanstd(np.stack(res_dic3_all),axis=0)   
heat_data3_mean = np.nanmean(np.stack(res_dic3_all),axis=0)


with open(folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_std.pkl', 'wb') as f:
    pickle.dump(heat_data0_std, f)
with open(folder_h+'/heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl', 'wb') as f:
    pickle.dump(heat_data0_mean, f)

with open(folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_std.pkl', 'wb') as f:
    pickle.dump(heat_data1_std, f)
with open(folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl', 'wb') as f:
    pickle.dump(heat_data1_mean, f)  
    
with open(folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_std.pkl', 'wb') as f:
    pickle.dump(heat_data2_std, f)
with open(folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl', 'wb') as f:
    pickle.dump(heat_data2_mean, f)

with open(folder_h+'heatmap_pred_phil_mimic4_run8_alt_rmse'+str(max_horizon)+'_seed_all_data_std.pkl', 'wb') as f:
    pickle.dump(heat_data3_std, f)
with open(folder_h+'heatmap_pred_phil_mimic4_run8_alt_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl', 'wb') as f:
    pickle.dump(heat_data3_mean, f)      


dataset='mimic4'
loss ='rmse'
save_heat= True
max_horizon=12
num_time = max_horizon
invert = True
colorbar=True
run=8

#mean
res_dic0_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5.png', bbox_inches='tight')  

res_dic1_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5.png', bbox_inches='tight')  

res_dic2_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5.png', bbox_inches='tight')  

res_dic3_mean= utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5.png', bbox_inches='tight')  


#std
res_dic0_std = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_std.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_std5.png', bbox_inches='tight')  

res_dic1_std = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_std.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_std5.png', bbox_inches='tight')  

res_dic2_std = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_std.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_std5.png', bbox_inches='tight')  

res_dic3_std = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_std.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_std5.png', bbox_inches='tight')  

# ###

#mean
vmin=0
vmax=2.5
res_dic0_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl_2_5.png', bbox_inches='tight')  

res_dic1_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl_2_5.png', bbox_inches='tight')  

res_dic2_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl_2_5.png', bbox_inches='tight')  

res_dic3_mean= utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl_2_5.png', bbox_inches='tight')  
####


###

save_heat = True
vmin=0
vmax=1.5
res_dic0_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl_1_5.png', bbox_inches='tight')  

res_dic1_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl_1_5.png', bbox_inches='tight')  

res_dic2_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl_1_5.png', bbox_inches='tight')  

res_dic3_mean= utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl_1_5.png', bbox_inches='tight')  


vmin=0
vmax=1.0
res_dic0_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl_1.png', bbox_inches='tight')  

res_dic1_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl_1.png', bbox_inches='tight')  

res_dic2_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl_1.png', bbox_inches='tight')  

res_dic3_mean= utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl_1.png', bbox_inches='tight')  
####

##paper
vmin=0
vmax=1.5
heat_max = 9
res_dic0_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='SOFA ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_SOFA_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar,heat_max=heat_max)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl_new1.5.png', bbox_inches='tight')  

res_dic1_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Creatinine ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_creatinine_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar,heat_max=heat_max)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl_new1.5.png', bbox_inches='tight')  

res_dic2_mean = utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='Bilirubin Total ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_bilirubin_total_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar,heat_max=heat_max)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl_new1.5.png', bbox_inches='tight')  

res_dic3_mean= utils_paper.heatmap_pred_dec(model, model_decoder, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, title ='ALT ',load_map=folder_h+'heatmap_pred_phil_mimic4_run8_ALT_rmse'+str(max_horizon)+'_seed_all_data_mean.pkl',vmin=vmin,vmax=vmax,invert=invert,colorbar=colorbar,heat_max=heat_max)
if save_heat: plt.savefig(folder_h+'heatmap_pred_phil_'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl_new1.5.png', bbox_inches='tight')  


###
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
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,2)),np.flipud(np.round(res_dic1_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line)) 
    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic2_mean.T,2)),np.flipud(np.round(res_dic2_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic3_mean.T,2)),np.flipud(np.round(res_dic3_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))     


#'/Users/jaschob/Desktop/mimic4_heat/OptAB_mimic4_seed/MimicIV_new_24/'