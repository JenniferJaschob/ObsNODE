#%%
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

import os
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Model_observableNODE_paper')
from utils_NODE import  LSTMRecognitionModel_with_nan, MyNeuralODE_withNNs, MyModel_adjoint, predict,heatmap_pred
from utils_NODE import *
num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

####
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#%%
folder_h ='/Users/jaschob/Desktop/Out_NODE/heat/'
path_data = '/Users/jaschob/Desktop/Paper1/semi_syn_mimic4/'
dataset = 'semisynmimicIV'


run = 1

if num_seed_i  > 100: 
    #obsNODE seed 101,102,103,104,105
    params_dic = {'n_z': 2,
                    'batch_size': 100,
                    'learning_rate': 0.001,
                    'hidden_dim_obs': 64,
                    'hidden_dim_node': 64,
                    'num_layers_node': 3,
                    'activation_node': 'leakyrelu'}
else: 
    #OLSN seed 1,2,3,4,5
    params_dic = {'n_z': 4,
                    'batch_size': 500,
                    'learning_rate': 0.001,
                    'hidden_dim_obs': 128,
                    'hidden_dim_node': 64,
                    'num_layers_node': 10,
                    'activation_node': 'tanh'}

n_z, n_x, n_u, n_t = params_dic['n_z'],2,2,0
    
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
list_index_t_s = [12,24,36]

#%%
# define Model ################################################################
myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
      
myNODE = MyNeuralODE_withNNs( activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
model = MyModel_adjoint(myNODE, myObserver).to(device)

    ####
read_path = '/Users/jaschob/Desktop/Out_NODE/model_state_dict_seed_'+str(num_seed_i)+'_semisynmimicIV.pth'

    #'/Users/jaschob/Desktop/Out_NODE/out_seed_'+str(num_seed_i)+'_cancer.pth'
model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))
#%%
# ##############################################################################
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

#%%
#
t_validate2, x_validate2 =    t_validate, x_validate
t_train2, x_train2 =          t_train, x_train
t_test2, x_test2 =            t_test, x_test
u_validate2, u_test2, u_train2 = u_validate, u_test, u_train

num_time= 48 
t_validate, x_validate =    t_validate2[:num_time], x_validate2[:num_time] 
t_train, x_train =          t_train2[:num_time], x_train2[:num_time]
t_test, x_test =            t_test2[:num_time], x_test2[:num_time]
u_validate, u_test, u_train = u_validate2[:num_time], u_test2[:num_time], u_train2[:num_time]   

#normalize = True
#if normalize:
#    y_train, y_validate, y_test = normalize_data(x_train, x_train), normalize_data(x_validate, x_train), normalize_data(x_test, x_train)
#    u_train, u_validate, u_test = normalize_data(u_train, u_train), normalize_data(u_validate, u_train), normalize_data(u_test, u_train)

    
#%%
index_t_s_i = 10
t_test_before, t_test_after     = t_test[:index_t_s_i].detach().clone(), t_test[index_t_s_i:].detach().clone()
x_test_before, x_test_after     = x_test[:index_t_s_i].detach().clone(), x_test[index_t_s_i:].detach().clone()
u_test_before, u_test_after     = u_test[:index_t_s_i].detach().clone(), u_test[index_t_s_i:].detach().clone()
        
x_pred_i, _, _,_  = predict(model=model,step_size=step_size,time_obs_pred=time_obs_pred,
                                              t_before=t_test_before,
                                              x_before=x_test_before,
                                              t_after=t_test_after,
                                              x_after=x_test_after,
                                              method=method,
                                              u_before=u_test_before,
                                              index_t_s=index_t_s_i,
                                              n_x_dach=n_x_dach,
                                              w=w)

#%%
num = 1
list_index_t_s_pred = list(range(1,x_train.size(0),num))
step = 1
loss = 'rmse'
max_horizon = len(list_index_t_s_pred) 
st=True
save_heat = True
save_res = True
dataset = 'semisynmimicIV'
pred_func = 'step'


save_res0 = folder_h+'heatmap_pred_'+dataset+'_run'+str(run)+'_tumorvolum_'+loss+'_horizon_'+str(max_horizon)
save_res1 = folder_h+'heatmap_pred_'+dataset+'_run'+str(run)+'_weight_'+loss+'_horizon_'+str(max_horizon)

if num_seed_i > 100:
    list_num_seed = [101,102,103,104,105] #[101,102,103,104,105] #[1,2,3,4,5]  
else: 
    list_num_seed = [1,2,3,4,5]
res_dic0_all, res_dic1_all = [], []

for seed_i in range(len(list_num_seed)):        
    num_seed_i = list_num_seed[seed_i]
    
    torch.manual_seed(num_seed_i)
    np.random.seed(num_seed_i)

    read_path = '/Users/jaschob/Desktop/Out_NODE/model_state_dict_seed_'+str(num_seed_i)+'_semisynmimicIV.pth'

    model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))

    res_dic0 = heatmap_pred(x_test,t_test,u_test,dataset=dataset, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=0, offset=0, max_horizon=max_horizon,loss=loss, pred_func=pred_func,title=' Synthetic Outcome ')#,vmin=0,vmax=0.9)   
    if save_heat: plt.savefig(save_res0 +'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight') 
    if save_res:
        with open(save_res0 +'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
            pickle.dump(res_dic0, f)
            
            
    res_dic0_all.append(res_dic0) 
#%%
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
res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =' Synthetic Outcome ',dataset=dataset,load_map=save_res0 +'_mean.pkl')
if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
if save_heat: plt.savefig(save_res0 +'_mean.pdf', bbox_inches='tight')  
res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = ' Synthetic Volum ',dataset=dataset,load_map=save_res0 +'_std.pkl')
if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  
if save_heat: plt.savefig(save_res0 +'_std.pdf', bbox_inches='tight') 


res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =' Synthetic Outcome ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
if save_heat: plt.savefig(save_res0 +'_mean1.pdf', bbox_inches='tight')  
# res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = ' Synthetic Volum ',dataset=dataset,load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.25)
# if save_heat: plt.savefig(save_res0 +'_std2.png', bbox_inches='tight')  


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
    

# %%
