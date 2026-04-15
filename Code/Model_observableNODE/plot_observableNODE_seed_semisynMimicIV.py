import torch
import numpy as np
import pickle 

from utils_observableNODE import *

import matplotlib.pyplot as plt


num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save = '/Users/jaschob/Desktop/semi_syn_mimic4/'+'/out_hyperopt_test'+str(num_seed_i)
folder = '/Users/jaschob/Desktop/semi_syn_mimic4/'

folder_h = '/Users/jaschob/Desktop/mimic4_heat/semisyn_mimic4/'
    

path_data = '/Users/jaschob/Desktop/semi_syn_mimic4/'
with open(path_data + "x_validate_syn_treat.pkl", "rb") as datei:
    x_validate = pickle.load(datei)        
with open(path_data + "x_train_syn_treat.pkl", "rb") as datei:
    x_train = pickle.load(datei)    
with open(path_data + "x_test_syn_treat.pkl", "rb") as datei:
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
    

num_time = 24
    
t_validate, x_validate =    t_validate[:num_time].to(torch.float).to(device), x_validate[:num_time].to(torch.float).to(device)
t_train, x_train =          t_train[:num_time].to(torch.float).to(device), x_train[:num_time].to(torch.float).to(device)
t_test, x_test =            t_test[:num_time].to(torch.float).to(device), x_test[:num_time].to(torch.float).to(device)


u_train, u_test, u_validate =  u_train[:num_time].to(torch.float).to(device),  u_test[:num_time].to(torch.float).to(device), u_validate[:num_time].to(torch.float).to(device)

s

params_dic = {'n_z': 2, 
              'batch_size': 100,
              'learning_rate': 0.0001,
              'hidden_dim_obs': 128,
              'hidden_dim_node': 128,
              'num_layers_node': 6, 
              'activation_node': 'leakyrelu'}

n_z = x_test.size(2)
n_x = x_test.size(2)
n_u = u_train.size(-1)
n_t = 0
    
n_x_dach = 2
w=0
    
# Trainingsparameter setzen
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
    list_index_t_s= [6,12,24,36]
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

read_path = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_semisynMimicIV/model_state_dict_seed_'+str(num_seed_i)+'_semisynMimicIV.pth' 
model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))

###
num = 1
list_index_t_s_pred = list(range(2,x_train.size(0),num))
step = 2
loss = 'rmse'
max_horizon = len(list_index_t_s_pred) 
st=True#False#True
save_heat = True
save_res =True
dataset = 'semisynMimic4'
pred_func = 'step'


save_res0 = folder_h+'heatmap_pred_'+dataset+'_val0_'+loss+'_horizon_'+str(max_horizon)

list_num_seed = [1,2,3,4,5]    
res_dic0_all, res_dic1_all = [], []

for seed_i in range(len(list_num_seed)):        
    num_seed_i = list_num_seed[seed_i]
    
    torch.manual_seed(num_seed_i)
    np.random.seed(num_seed_i)
    
    read_path = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_semisynMimicIV/model_state_dict_seed_'+str(num_seed_i)+'_semisynMimicIV.pth' 
    model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))

    res_dic0 = heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=0, offset=0, max_horizon=max_horizon,loss=loss, pred_func=pred_func,title=' Value 0 ')#,vmin=0,vmax=0.9)   
    if save_heat: plt.savefig(save_res0 +'_seed_'+str(num_seed_i)+'.png', bbox_inches='tight') 
    if save_res:
        with open(save_res0 +'_seed_'+str(num_seed_i)+'.pkl', 'wb') as f:
            pickle.dump(res_dic0, f)
            
            
    res_dic0_all.append(res_dic0) 
    

heat_data0_std = np.nanstd(np.stack(res_dic0_all),axis=0)   
heat_data0_mean = np.nanmean(np.stack(res_dic0_all),axis=0)      
with open(save_res0 +'_seed_all.pkl', 'wb') as f:
    pickle.dump(res_dic1_all, f)
with open(save_res0 +'_mean.pkl', 'wb') as f:
    pickle.dump(heat_data0_mean, f)    
with open(save_res0 +'_std.pkl', 'wb') as f:
    pickle.dump(heat_data0_std, f)    


####
res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =' Value 0 ',dataset=dataset,load_map=save_res0 +'_mean.pkl')
if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = ' Value 0 ',dataset=dataset,load_map=save_res0 +'_std.pkl')
if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  



# res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =' Value 0',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
# if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
# res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = ' Value 0 ',dataset=dataset,load_map=save_res0 +'_std.pkl',vmin=0,vmax=0.1)
# if save_heat: plt.savefig(save_res0 +'_std1.png', bbox_inches='tight')  

vmin=0
vmax=1.0
save_heat=True
heat_max = 24
res_dic0 = '/Users/jaschob/Desktop/mimic4_heat/causalNODE_semisyn_mimic4/use/heatmap_pred_semi_syn_mimic4_run8_val0_rmse_horizon_23'

res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title =' Value 0 ',dataset=dataset,load_map=res_dic0 +'_mean.pkl',vmax=vmax,vmin=vmin,heat_max=heat_max)
if save_heat: plt.savefig(res_dic0 +'_mean_'+str(vmax)+'.png', bbox_inches='tight')  
res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = ' Value 0 ',dataset=dataset,load_map=res_dic0 +'_std.pkl')
if save_heat: plt.savefig(res_dic0 +'_std.png', bbox_inches='tight')  


with open(res_dic0 +'_mean.pkl' , 'rb') as handle:
    res_dic0_mean = pickle.load(handle)
with open(res_dic0 +'_std.pkl', 'rb') as handle:
    res_dic0_std = pickle.load(handle)
    
    

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
    



