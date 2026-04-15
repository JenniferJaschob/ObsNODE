import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils_observableNODE import *

num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

print(torch.cuda.is_available())
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
t_validate, x_validate =    t_validate[:,:,val_list].to(torch.float).to(device), x_validate[:,:,val_list].to(torch.float).to(device)
t_train, x_train =          t_train[:,:,val_list].to(torch.float).to(device), x_train[:,:,val_list].to(torch.float).to(device)
t_test, x_test =            t_test[:,:,val_list].to(torch.float).to(device), x_test[:,:,val_list].to(torch.float).to(device)
    
    
u_train[torch.isnan(u_train)],u_test[torch.isnan(u_test)],u_validate[torch.isnan(u_validate)] = 0,0,0

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

list_index_t_s = [4,6,8,10]

# define Model ################################################################
myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
    
helperNN_list = [] 
for j in range(1,n_z):
    helperNN= MyhelperNN(activation = activation_helperNN ,hidden_sizes =hidden_sizes_helperNN ,num_layers=num_layers_helperNN,n_u=n_u,n_z_i=j*n_x,output_size=n_x).to(device)
    helperNN_list.append(helperNN)
        
myNODE = MyObservableNeuralODE_withNNs(helperNN_list=helperNN_list, activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
model = MyModel_adjoint(myNODE, myObserver).to(device)

read_path = '/Users/jaschob/Desktop/Paper_Code/Model_observableNODE/out/out_MimicV/model_state_dict_seed_'+str(num_seed_i)+'_MimicIV.pth' 
model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))


# Validierung ##################################################################
t_validate2, x_validate2 =    t_validate, x_validate
t_train2, x_train2 =          t_train, x_train
t_test2, x_test2 =            t_test, x_test
u_validate2, u_test2, u_train2 = u_validate, u_test, u_train


num_time=24
t_validate, x_validate =    t_validate2[:num_time], x_validate2[:num_time] 
t_train, x_train =          t_train2[:num_time], x_train2[:num_time]
t_test, x_test =            t_test2[:num_time], x_test2[:num_time]
u_validate, u_test, u_train = u_validate2[:num_time], u_test2[:num_time], u_train2[:num_time]     


num = 1
list_index_t_s_pred = list(range(1,x_train.size(0),num))
step = 2
loss = 'rmse'
max_horizon = len(list_index_t_s_pred) 
st=True
save_heat = False
save_res = False
folder_h = '/Users/jaschob/Desktop/mimic4_heat/test/'#'/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/'     
folder_h = '/Users/jaschob/Desktop/mimic4_heat/causalNODE_mimic4_seed/horizon10/'
dataset = 'mimic4'
pred_func = 'step'
step_pred = True
step_size = 1.0
data_step = 2.0
run = 8
stop
res_dic0 = heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=0, offset=0, max_horizon=max_horizon,loss=loss,dataset = dataset, pred_func=pred_func, title = 'SOFA-Score ',step_pred=step_pred,data_step=data_step)#,vmin=0,vmax=0.7)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'_step.png', bbox_inches='tight') 
if save_res: 
    with open(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'data_step.pkl', 'wb') as f:
        pickle.dump(res_dic0, f)
    
res_dic1 = heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=1, offset=0, max_horizon=max_horizon,loss=loss,dataset = dataset, pred_func=pred_func, title = 'Creatinine ',step_pred=step_pred,data_step=data_step)#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'_step.png', bbox_inches='tight')     
if save_res:
    with open(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'data_step.pkl', 'wb') as f:
        pickle.dump(res_dic1, f)
        
res_dic2 = heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=2, offset=0, max_horizon=max_horizon,loss=loss,dataset = dataset, pred_func=pred_func,  title = 'Bilirubin total ',step_pred=step_pred,data_step=data_step)#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'_step.png', bbox_inches='tight') 
if save_res:
    with open(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'data_step.pkl','wb') as f:
        pickle.dump(res_dic2, f)
        
res_dic3 = heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred=list_index_t_s_pred,model=model,n_x_dach=n_x_dach, w=w,step_size=step_size,time_obs_pred=time_obs_pred,method=method,st=st,x_train=x_train,step = step,index=3, offset=0, max_horizon=max_horizon,loss=loss,dataset = dataset, pred_func=pred_func,  title ='ALT ',step_pred=step_pred,data_step=data_step)#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'_step.png', bbox_inches='tight')     
if save_res:
    with open(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_'+str(num_seed_i)+'data_step.pkl', 'wb') as f:
        pickle.dump(res_dic3, f)

res_dic0_all, res_dic1_all,res_dic2_all, res_dic3_all = [], [], [], []
for seed in range(2,7):

    with open(folder_h+'heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_'+str(seed)+'data_step.pkl', 'rb') as f:
        res_dic0 = pickle.load(f)
        
    with open(folder_h+'heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_'+str(seed)+'data_step.pkl', 'rb') as f:
        res_dic1 = pickle.load(f)

    with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_'+str(seed)+'data_step.pkl', 'rb') as f:
        res_dic2 = pickle.load(f)

    with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_'+str(seed)+'data_step.pkl', 'rb') as f:
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


with open(folder_h+'heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_std_step.pkl', 'wb') as f:
    pickle.dump(heat_data0_std, f)
with open(folder_h+'heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl', 'wb') as f:
    pickle.dump(heat_data0_mean, f)

with open(folder_h+'heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_std_step.pkl', 'wb') as f:
    pickle.dump(heat_data1_std, f)
with open(folder_h+'heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl', 'wb') as f:
    pickle.dump(heat_data1_mean, f)  
    
with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_std_step.pkl', 'wb') as f:
    pickle.dump(heat_data2_std, f)
with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl', 'wb') as f:
    pickle.dump(heat_data2_mean, f)

with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_std_step.pkl', 'wb') as f:
    pickle.dump(heat_data3_std, f)
with open(folder_h+'mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl', 'wb') as f:
    pickle.dump(heat_data3_mean, f)      


# t_validate2, x_validate2 =    t_validate, x_validate
# t_train2, x_train2 =          t_train, x_train
# t_test2, x_test2 =            t_test, x_test
# u_validate2, u_test2, u_train2 = u_validate, u_test, u_train


# num_time=24
# t_validate, x_validate =    t_validate2[:num_time], x_validate2[:num_time] 
# t_train, x_train =          t_train2[:num_time], x_train2[:num_time]
# t_test, x_test =            t_test2[:num_time], x_test2[:num_time]
# u_validate, u_test, u_train = u_validate2[:num_time], u_test2[:num_time], u_train2[:num_time] 

# os.chdir('/Users/jaschob/Documents/GitHub/Promotion1_Jennifer')
# from  my_vergleichsmasse import my_compute_prediction,heatmap_pred
# folder_h = '/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/'
# dataset='mimic4'
# loss ='rmse'
# save_heat= False


res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=0.9)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl_step.png', bbox_inches='tight')  

res_dic1_mean = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=1.5)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl_step.png', bbox_inches='tight')  

res_dic2_mean = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=2.6)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl_step.png', bbox_inches='tight')  

res_dic3_mean = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=2.1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl_step.png', bbox_inches='tight')  

res_dic0_std = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.2)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_std5_vgl_step.png', bbox_inches='tight')  

res_dic1_std = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.2)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_std5_vg_stepl.png', bbox_inches='tight')  

res_dic2_std = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.2)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_std5_vgl_step.png', bbox_inches='tight')  

res_dic3_std = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.3)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_std5_vgl_step.png', bbox_inches='tight')  

####
res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_vgl1_step.png', bbox_inches='tight')  

res_dic1_mean = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_vgl1_step.png', bbox_inches='tight')  

res_dic2_mean = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_vgl1_step.png', bbox_inches='tight')  

res_dic3_mean = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_vgl1_step.png', bbox_inches='tight')  

res_dic0_std = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.3)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_std5_vgl1_step.png', bbox_inches='tight')  

res_dic1_std = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.3)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_std5_vgl1_step.png', bbox_inches='tight')  

res_dic2_std = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.3)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_std5_vgl1_step.png', bbox_inches='tight')  

res_dic3_std = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_std_step.pkl',vmin=0,vmax=0.3)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_std5_vgl1_step.png', bbox_inches='tight')  

 
###

res_dic0_mean = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_mean5_step.png', bbox_inches='tight')  

res_dic1_mean = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_mean5_step.png', bbox_inches='tight')  

res_dic2_mean = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_mean5_step.png', bbox_inches='tight')  

res_dic3_mean = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_mean_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_mean5_step.png', bbox_inches='tight')  



res_dic0_std = heatmap_pred(x_test,t_test,u_test, title ='SOFA ',load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_SOFA_rmse'+str(num_time)+'_seed_all_data_std_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_SOFA_'+loss+str(num_time)+'_seed_std5.png', bbox_inches='tight')  

res_dic1_std = heatmap_pred(x_test,t_test,u_test, title ='Creatinine ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_creatinine_rmse'+str(num_time)+'_seed_all_data_std_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_creatinine_'+loss+str(num_time)+'_seed_std5_step.png', bbox_inches='tight')  

res_dic2_std = heatmap_pred(x_test,t_test,u_test, title ='Bilirubin total ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_bilirubin_total_rmse'+str(num_time)+'_seed_all_data_std_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_bilirubin_total_'+loss+str(num_time)+'_seed_std5_step.png', bbox_inches='tight')  

res_dic3_std = heatmap_pred(x_test,t_test,u_test, title ='ALT ',dataset=dataset,load_map='/Users/jaschob/Desktop/mimic4_heat/mimic4_seed/heatmap_predmimic4_run8_alt_rmse'+str(num_time)+'_seed_all_data_std_step.pkl')#,vmin=0,vmax=1)
if save_heat: plt.savefig(folder_h+'heatmap_pred'+dataset+'_run'+str(run)+'_alt_'+loss+str(num_time)+'_seed_std5_step.png', bbox_inches='tight')  
##


####

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
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,3)),np.flipud(np.round(res_dic1_std.T,3))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.3f}'
            s_str = f'{s:.3f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line)) 
    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic2_mean.T,3)),np.flipud(np.round(res_dic2_std.T,3))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.3f}'
            s_str = f'{s:.3f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic3_mean.T,3)),np.flipud(np.round(res_dic3_std.T,3))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.3f}'
            s_str = f'{s:.3f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))     
