#%%
import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils

import matplotlib.pyplot as plt
import pickle


print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#%%
#data 
gamma = 4
print(gamma)
transformed_datapath =  '/Users/jaschob/Desktop/DoseAI/data/data_dict_'+str(gamma)+'_'+str(gamma)+'.p'
model_path = '/Users/jaschob/Desktop/Paper1/Paper_Code/Model_OptAB_DoseAI/train_pth/train_pth_cancer/'
pickle_map = read_from_file(transformed_datapath)
training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)


thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])    

    
#params    
treatment_options = 2
 
hidden_channels = 16
batch_size = 250
hidden_states = 578
lr = 0.004239690693777566
activation = 'leakyrelu'
num_depth = 2
pred_act = 'tanh'
pred_states = 128
pred_depth = 4
pred_comp=True 
    

offset=0
max_horizon = 3
    
input_channels_dec=3
output_channels=2


hidden_channels_dec = 22
batch_size_dec = 125
hidden_states_dec = 802
lr_dec = 0.0016227982436909543
activation_dec = 'leakyrelu'
num_depth_dec = 13
pred_act_dec = 'leakyrelu'
pred_states_dec = 798
pred_depth_dec = 1

#%%
folder_h = '/Users/jaschob/Desktop/paper1/mimic4_heat/doseai_cancer_seed/gamma'+str(gamma)+'/'

save_res0 = folder_h+'heatmap_pred_DoseAI_tumorvolum_horizon_'+str(max_horizon)+'gamma'+str(gamma)
save_res1 = folder_h+'heatmap_pred_DoseAI_weight_horizon_'+str(max_horizon)+'gamma'+str(gamma)


dataset = dataset = 'DoseAI'
invert = False
save_heat = True

    #data
if training_processed is not None:
    train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, device)
if validation_processed is not None:
    validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,device)
if test_processed is not None:
    test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test = utils.prep_map(test_processed,device) 
#save_res = True
    
list_num_seed = [0,1,2,3,4]    
res_dic0_all, res_dic1_all = [], []
# stop
# for seed_i in range(len(list_num_seed)):        
#     num_seed_i = list_num_seed[seed_i]
    
#     print(num_seed_i)
#     torch.manual_seed(num_seed_i)
#     np.random.seed(num_seed_i)
        
    
#     save = str(model_path+'Trained_Encoder_seed_'+str(num_seed_i)+'_cancer.pth')   
#     save2 = str(model_path+'Trained_Decoder_seed_'+str(num_seed_i)+'_cancer.pth')
    
#     model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)
#     model.load_state_dict(torch.load(save))
#     model=model.to(model.device)
        
#     model_decoder = utils.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec)
#     model_decoder.load_state_dict(torch.load(save2))
#     model_decoder=model_decoder.to(model.device)
    
    
#     offset=0
#     max_horizon=12
    
#     #data
#     if training_processed is not None:
#         train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
#     if validation_processed is not None:
#         validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
#     if test_processed is not None:
#         test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test = utils.prep_map(test_processed,model.device) 

#     #test
#     res_dic0 = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=0, offset=offset, max_horizon=max_horizon, loss='rmse',invert=invert, title=str(gamma)+ ' Tumor Volum')#,vmin=0,vmax=0.6)
#     if save_heat: plt.savefig(save_res0+'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight')
#     if save_res:
#         with open(save_res0+'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
#             pickle.dump(res_dic0, f)
        
    
#     res_dic1 = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, index=1, offset=offset, max_horizon=max_horizon, loss='rmse',invert=invert, title =str(gamma)+ ' Wight ')#,vmin=0,vmax=0.9)        
#     if save_heat: plt.savefig(save_res1+'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight')
#     if save_res:
#         with open(save_res1+'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
#             pickle.dump(res_dic1, f)     
            
            
#     res_dic0_all.append(res_dic0) 
#     res_dic1_all.append(res_dic1) 
     
 
# heat_data0_std = np.nanstd(np.stack(res_dic0_all),axis=0)   
# heat_data0_mean = np.nanmean(np.stack(res_dic0_all),axis=0)      
# with open(save_res0 +'_seed_all.pkl', 'wb') as f:
#      pickle.dump(res_dic1_all, f)
# with open(save_res0 +'_mean.pkl', 'wb') as f:
#      pickle.dump(heat_data0_mean, f)    
# with open(save_res0 +'_std.pkl', 'wb') as f:
#      pickle.dump(heat_data0_std, f)    


# heat_data1_std = np.nanstd(np.stack(res_dic1_all),axis=0)   
# heat_data1_mean = np.nanmean(np.stack(res_dic1_all),axis=0)
# with open(save_res1 +'_seed_all.pkl', 'wb') as f:
#      pickle.dump(res_dic1_all, f)
# with open(save_res1 +'_mean.pkl', 'wb') as f:
#      pickle.dump(heat_data1_mean, f) 
# with open(save_res1 +'_std.pkl', 'wb') as f:
#      pickle.dump(heat_data1_std, f)             


# res_dic0_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Tumor Volum ',load_map=save_res0 +'_mean.pkl')
# if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
# res_dic0_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title = str(gamma)+ ' Tumor Volum ',load_map=save_res0 +'_std.pkl')
# if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  

# res_dic1_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_mean.pkl')
# if save_heat: plt.savefig(save_res1 +'_mean.png', bbox_inches='tight')      
# res_dic1_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_std.pkl')
# if save_heat: plt.savefig(save_res1 +'_std.png', bbox_inches='tight') 
# ###


# res_dic0_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Tumor Volum ',load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
# res_dic0_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title = str(gamma)+ ' Tumor Volum ',load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.1)
# if save_heat: plt.savefig(save_res0 +'_std1.png', bbox_inches='tight')  

# res_dic1_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
# res_dic1_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_std.pkl',vmin=0,vmax=0.1)
# if save_heat: plt.savefig(save_res1 +'_std1.png', bbox_inches='tight') 


# ####
# res_dic0_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Tumor Volum ',load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
# res_dic0_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title = str(gamma)+ ' Tumor Volum ',load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.1)
# if save_heat: plt.savefig(save_res0 +'_std1.png', bbox_inches='tight')  

# res_dic1_mean = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
# res_dic1_std = utils.heatmap_pred_dec(model, model_decoder, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_std.pkl',vmin=0,vmax=0.1)
# if save_heat: plt.savefig(save_res1 +'_std1.png', bbox_inches='tight') 

#%%
save_heat = False
res_dic0_mean = utils.heatmap_pred_dec(None,None,test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Tumor Volume ',load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
res_dic0_std = utils.heatmap_pred_dec(None,None, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title = str(gamma)+ ' Tumor Volume ',load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.1)
if save_heat: plt.savefig(save_res0 +'_std1.png', bbox_inches='tight')  

res_dic1_mean = utils.heatmap_pred_dec(None,None,test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_mean.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
res_dic1_std = utils.heatmap_pred_dec(None,None, test_processed,test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test, title =str(gamma)+' Weight ',load_map=save_res1 +'_std.pkl',vmin=0,vmax=0.1)
if save_heat: plt.savefig(save_res1 +'_std1.png', bbox_inches='tight') 


for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,2)),np.flipud(np.round(res_dic0_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  # leer lassen bei nan
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    

#Wight    
for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,2)),np.flipud(np.round(res_dic1_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  # leer lassen bei nan
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))      

# #Tumor volum
# for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,3)),np.flipud(np.round(res_dic0_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  # leer lassen bei nan
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))    

# #Wight    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,3)),np.flipud(np.round(res_dic1_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  # leer lassen bei nan
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))      
# %%
