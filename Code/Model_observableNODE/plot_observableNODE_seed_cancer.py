#%%
import torch
import numpy as np

import pickle

import matplotlib.pyplot as plt
#from utils_observableNODE import *


num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

####
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

####
gamma=4

folder_h ='/Users/jaschob/Desktop/mimic4_heat/causalNODE_cancer_seed/gamma'+str(gamma)+'/'
path_data = '/Users/jaschob/Desktop/Paper_Code/Data/Cancer/dataset_cancer/'#"/Users/jaschob/Desktop/DoseAI/data/"

dataset = 'DoseAI'


run = 1

#hyperopt
params_dic = {'n_z': 2,
     'batch_size': 64,
     'learning_rate': 0.0001,
     'hidden_dim_obs': 128,
     'hidden_dim_node': 128,
     'num_layers_node': 5,
     'activation_node': 'tanh'}


n_z, n_x, n_u, n_t = 2,2,2,0
    
n_x_dach = 2
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
list_index_t_s = [4, 8, 12, 16, 20]

#%%
# define Model ################################################################
myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
    
helperNN_list = [] 
for j in range(1,n_z):
    helperNN= MyhelperNN(activation = activation_helperNN ,hidden_sizes =hidden_sizes_helperNN ,num_layers=num_layers_helperNN,n_u=n_u,n_z_i=j*n_x,output_size=n_x).to(device)
    helperNN_list.append(helperNN)
        
myNODE = MyObservableNeuralODE_withNNs(helperNN_list=helperNN_list, activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
model = MyModel_adjoint(myNODE, myObserver).to(device)

####
read_path = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_cancer/model_state_dict_seed_'+str(num_seed_i)+'_cancer.pth' 
model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))


# ##############################################################################
#%%
## Data Gamma

if gamma != None:            
    
    with open(path_data+"data_val_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
         data_validate  = pickle.load(datei)
                        
    with open(path_data+"data_test_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
        data_test = pickle.load(datei)
                    
    with open(path_data+"data_train_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
        data_train = pickle.load(datei)
        
            
    with open(path_data+"data_val_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_validate  = pickle.load(datei)
                    
    with open(path_data+"data_test_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_test = pickle.load(datei)
                
    with open(path_data+"data_train_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_train = pickle.load(datei)           
        
        
    t_validate, x_validate, _  =  data_validate
    t_train, x_train, _        =  data_train
    t_test, x_test, _          =  data_test
        
    u_train[torch.isnan(u_train)],u_test[torch.isnan(u_test)],u_validate[torch.isnan(u_validate)] = 0,0,0


# Data für verschiedenen horizon    
t_validate2, x_validate2 =    t_validate, x_validate
t_train2, x_train2 =          t_train, x_train
t_test2, x_test2 =            t_test, x_test
u_validate2, u_test2, u_train2 = u_validate, u_test, u_train

num_time=12+4
t_validate, x_validate =    t_validate2[:num_time], x_validate2[:num_time] 
t_train, x_train =          t_train2[:num_time], x_train2[:num_time]
t_test, x_test =            t_test2[:num_time], x_test2[:num_time]
u_validate, u_test, u_train = u_validate2[:num_time], u_test2[:num_time], u_train2[:num_time]         
    



num = 1
list_index_t_s_pred = list(range(4,x_train.size(0),num))
step = 1
loss = 'rmse'
max_horizon = len(list_index_t_s_pred) 
st=True#False#True
save_heat = False
save_res =False
dataset = 'DoseAI'
pred_func = 'step'


save_res0 = folder_h+'heatmap_pred_'+dataset+'_run'+str(run)+'_tumorvolum_'+loss+'_horizon_'+str(max_horizon)+'gamma'+str(gamma)
save_res1 = folder_h+'heatmap_pred_'+dataset+'_run'+str(run)+'_weight_'+loss+'_horizon_'+str(max_horizon)+'gamma'+str(gamma)


# list_num_seed = [1,2,3,4,5]  
# res_dic0_all, res_dic1_all = [], []

# for seed_i in range(len(list_num_seed)):        
#     num_seed_i = list_num_seed[seed_i]
    
#     torch.manual_seed(num_seed_i)
#     np.random.seed(num_seed_i)
#     read_path = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_cancer/model_state_dict_seed_'+str(num_seed_i)+'_cancer.pth' 
#     model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))

#     res_dic0 = heatmap_pred(x_test,t_test,u_test,dataset=dataset, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=0, offset=0, max_horizon=max_horizon,loss=loss, pred_func=pred_func,title=str(gamma)+' Tumor volum ')#,vmin=0,vmax=0.9)   
#     if save_heat: plt.savefig(save_res0 +'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight') 
#     if save_res:
#         with open(save_res0 +'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
#             pickle.dump(res_dic0, f)
            
#     res_dic1 = heatmap_pred(x_test,t_test,u_test, dataset=dataset,list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=1, offset=0, max_horizon=max_horizon,loss=loss, pred_func=pred_func,title=str(gamma)+' Wight ')#,vmin=0,vmax=0.6)
#     if save_heat: plt.savefig(save_res1 +'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight')     
#     if save_res:
#         with open(save_res1 +'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
#             pickle.dump(res_dic1, f)
            
#     res_dic0_all.append(res_dic0) 
#     res_dic1_all.append(res_dic1) 
# stop
# heat_data0_std = np.nanstd(np.stack(res_dic0_all),axis=0)   
# heat_data0_mean = np.nanmean(np.stack(res_dic0_all),axis=0)      
# with open(save_res0 +'_seed_all.pkl', 'wb') as f:
#     pickle.dump(res_dic1_all, f)
# with open(save_res0 +'_mean.pkl', 'wb') as f:
#     pickle.dump(heat_data0_mean, f)    
# with open(save_res0 +'_std.pkl', 'wb') as f:
#     pickle.dump(heat_data0_std, f)    


# heat_data1_std = np.nanstd(np.stack(res_dic1_all),axis=0)   
# heat_data1_mean = np.nanmean(np.stack(res_dic1_all),axis=0)
# with open(save_res1 +'_seed_all.pkl', 'wb') as f:
#     pickle.dump(res_dic1_all, f)
# with open(save_res1 +'_mean.pkl', 'wb') as f:
#     pickle.dump(heat_data1_mean, f) 
# with open(save_res1 +'_std.pkl', 'wb') as f:
#     pickle.dump(heat_data1_std, f)   


####
save_heat = False
res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_mean.pkl')
if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_std.pkl')
if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  

res_dic1_mean = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 +'_mean.pkl')
if save_heat: plt.savefig(save_res1 +'_mean.png', bbox_inches='tight')      
res_dic1_std = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 +'_std.pkl')
if save_heat: plt.savefig(save_res1 +'_std.png', bbox_inches='tight') 
###


# res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
# res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.25)
# if save_heat: plt.savefig(save_res0 +'_std2.png', bbox_inches='tight')  

# res_dic1_mean = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
# res_dic1_std = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 +'_std.pkl',vmin=0,vmax=0.25)
# if save_heat: plt.savefig(save_res1 +'_std2.png', bbox_inches='tight') 

#print
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


# ####
# for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,3)),np.flipud(np.round(res_dic0_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))    
    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,3)),np.flipud(np.round(res_dic1_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))  




