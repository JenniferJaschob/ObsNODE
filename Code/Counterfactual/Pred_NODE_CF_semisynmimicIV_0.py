#%%
import torch
import numpy as np
import pickle
import joblib

import os
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Model_observableNODE_paper')
from utils_NODE import  LSTMRecognitionModel_with_nan, MyNeuralODE_withNNs, MyModel_adjoint, predict
from utils_NODE import *

num_seed_i = 105 101,... 105 obsNODE
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

####
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### set paths ###
path_data = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/SemisynMimicIV_CF/data_semisynmimicIV/CF/'
path_data2 = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/SemisynMimicIV_CF/data_semisynmimicIV/simulate/'
save_path = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/SemisynMimicIV_CF/Pred_CF_semisynmimicIV/NODE/'
read_path = '/Users/jaschob/Desktop/Out_NODE/' 
model_name = 'NODE'


#%%
for i in range(1, 6):
    cf = i   
    with open(path_data + "x_test_syn_"+str(cf)+".pkl", "rb") as datei:
        x_test = pickle.load(datei)
   
    with open(path_data + "u_test_syn_" + str(cf) + ".pkl", "rb") as datei:
        u_test = pickle.load(datei)   

    #default data      
    with open(path_data2 + "x_train_syn_treat.pkl", "rb") as datei:
        x_train = pickle.load(datei)    
    with open(path_data2 + "x_test_syn_treat.pkl", "rb") as datei:
        x_test_default = pickle.load(datei)
   
    with open(path_data2 + "u_train_syn.pkl", "rb") as datei:
        u_train = pickle.load(datei)
    with open(path_data2 + "u_test_syn.pkl", "rb") as datei:
        u_test_default = pickle.load(datei)    
    
    with open(path_data2 + "t_train.pkl", "rb") as datei:
        t_train = pickle.load(datei)
    with open(path_data2 + "t_test.pkl", "rb") as datei:
        t_test = pickle.load(datei)   

    
    u_test[torch.isnan(u_test)] = 0
    u_test_default[torch.isnan(u_test_default)] = 0
    u_train[torch.isnan(u_train)] = 0

    num_time = 48
    
    t_train, x_train =          t_train[:num_time].to(torch.float).to(device), x_train[:num_time].to(torch.float).to(device)
    t_test, x_test =            t_test[:num_time].to(torch.float).to(device), x_test[:num_time].to(torch.float).to(device)
    x_test_default =          x_test_default[:num_time].to(torch.float).to(device)
    u_train, u_test, u_test_default =  u_train[:num_time].to(torch.float).to(device),  u_test[:num_time].to(torch.float).to(device), u_test_default[:num_time].to(torch.float).to(device)


    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True)


    # define Parameters
    if num_seed_i  > 100: 
        params_dic = {'n_z': 2,
                'batch_size': 100,
                'learning_rate': 0.001,
                'hidden_dim_obs': 64,
                'hidden_dim_node': 64,
                'num_layers_node': 3,
                'activation_node': 'leakyrelu'}

    else:
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
    if num_time == 48:
        list_index_t_s= [6,12,24,36]
    elif num_time==12:
        list_index_t_s = [4,6,8,10]    
    else:   
        list_index_t_s = [4,6,8,10]


    # define Model ################################################################
    myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
      
    myNODE = MyNeuralODE_withNNs( activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
    model = MyModel_adjoint(myNODE, myObserver).to(device)

    ####
    model.load_state_dict(torch.load(read_path+'model_state_dict_seed_'+str(num_seed_i)+'_semisynMimicIV.pth',map_location=torch.device('cpu')))


    # ##############################################################################
    # Prediction
    #train test split
    #num_time = 24
    t_test, x_test, u_test =    t_test[:num_time], x_test[:num_time], u_test[:num_time] 

    index_t_s_i = 5
    t_test_before, t_test_after     = t_test[:index_t_s_i].detach().clone(), t_test[index_t_s_i:].detach().clone()
    x_test_before, x_test_after     = x_test[:index_t_s_i].detach().clone(), x_test[index_t_s_i:].detach().clone()
    u_test_before, u_test_after     = u_test[:index_t_s_i].detach().clone(), u_test[index_t_s_i:].detach().clone()
    
    x_test_before_default, x_test_after_default     = x_test_default[:index_t_s_i].detach().clone(), x_test_default[index_t_s_i:].detach().clone()

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
    

    # save
    
    with open(save_path+"data_pred_semisynMimicIV_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_NODE.pkl", "wb") as datei:
            pickle.dump(x_pred_i,datei)


    x_pred_i = (x_pred_i - mean) / std
    x_test_after = (x_test_after - mean) / std
    x_test_after_default = (x_test_after_default - mean) / std

    # save
    with open(save_path+"data_pred_semisynMimicIV_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_NODE.pkl", "wb") as datei:
            pickle.dump(x_pred_i,datei)

    diff_predcf_truecf = x_pred_i - x_test_after#y(ai)-y(ai)
    cb_per_serie_variable = diff_predcf_truecf.mean(dim=1)#CB_pred
    cb_per_variable = diff_predcf_truecf.mean(dim=(0, 1))

    diff_predcf_true = x_pred_i - x_test_after_default#y_pred(ai)-y(a0))
    ate_per_serie_variable = diff_predcf_true.mean(dim=1)#ATE_pred
    ate_per_variable = diff_predcf_true.mean(dim=(0, 1))

    diff_cf_true = x_test_after - x_test_after_default #y(ai)-y(a0)
    ate_per_serie_variable_default = diff_cf_true.mean(dim=1)#ATE 
    ate_per_variable_default = diff_cf_true.mean(dim=(0, 1))

    diff_ate = diff_predcf_true - diff_cf_true #y_pred(ai)-y(a0) - (y(ai)-y(a0))= y_pred(ai)-y(ai)
    diff_ate_per_serie_variable = diff_ate.mean(dim=1)#ATE_pred - ATE =! CB 
    diff_ate_per_variable = diff_ate.mean(dim=(0, 1))

    list = ['cb_per_serie_variable', 'cb_per_variable', 'ate_per_serie_variable', 'ate_per_variable', 'ate_per_serie_variable_default', 'ate_per_variable_default', 'diff_ate_per_serie_variable', 'diff_ate_per_variable']
    with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_"+model_name+".pkl", "wb") as datei:
        pickle.dump((cb_per_serie_variable, cb_per_variable, ate_per_serie_variable, ate_per_variable, ate_per_serie_variable_default, ate_per_variable_default, diff_ate_per_serie_variable, diff_ate_per_variable,list),datei)        


# %% 
if num_seed_i > 100:
      num_seed_range = range(101, 106)
else:
      num_seed_range = range(1, 6)

for cf in range(1, 6):
    
    cb_per_variable_list, cb_per_serie_variable_list  = [], []
    ate_per_variable_list, ate_per_serie_variable_list  = [], []
    diff_ate_per_variable_list, diff_ate_per_serie_variable_list = [], []

    for num_seed_i in num_seed_range:

        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_"+model_name+".pkl", "rb") as datei:
            cb_per_serie_variable, cb_per_variable, ate_per_serie_variable, ate_per_variable, ate_per_serie_variable_default, ate_per_variable_default, diff_ate_per_serie_variable, diff_ate_per_variable,list = pickle.load(datei)

        cb_per_variable_list.append(cb_per_variable)
        cb_per_serie_variable_list.append(cb_per_serie_variable)
        ate_per_variable_list.append(ate_per_variable)
        ate_per_serie_variable_list.append(ate_per_serie_variable)
        diff_ate_per_serie_variable_list.append(diff_ate_per_serie_variable)
        diff_ate_per_variable_list.append(diff_ate_per_variable)

    cb_per_variable_mean = torch.stack(cb_per_variable_list).mean(dim=0)
    cb_per_serie_variable_mean = torch.stack(cb_per_serie_variable_list).mean(dim=0)
    ate_per_variable_mean = torch.stack(ate_per_variable_list).mean(dim=0)
    ate_per_serie_variable_mean = torch.stack(ate_per_serie_variable_list).mean(dim=0) 
    diff_ate_per_variable_mean = torch.stack(diff_ate_per_variable_list).mean(dim=0)
    diff_ate_per_serie_variable_mean = torch.stack(diff_ate_per_serie_variable_list).mean(dim=0)

    cb_per_variable_std = torch.stack(cb_per_variable_list).std(dim=0)
    cb_per_serie_variable_std = torch.stack(cb_per_serie_variable_list).std(dim=0)
    ate_per_variable_std = torch.stack(ate_per_variable_list).std(dim=0)
    ate_per_serie_variable_std = torch.stack(ate_per_serie_variable_list).std(dim=0)    
    diff_ate_per_variable_std = torch.stack(diff_ate_per_variable_list).std(dim=0)
    diff_ate_per_serie_variable_std = torch.stack(diff_ate_per_serie_variable_list).std(dim=0)

    list_mean = ['cb_per_serie_variable_mean', 'cb_per_variable_mean', 'ate_per_serie_variable_mean', 'ate_per_variable_mean', 'ate_per_serie_variable_default_mean', 'ate_per_variable_default_mean', 'diff_ate_per_serie_variable_mean', 'diff_ate_per_variable_mean']
    list_std = ['cb_per_serie_variable_std', 'cb_per_variable_std', 'ate_per_serie_variable_std', 'ate_per_variable_std', 'ate_per_serie_variable_default_std', 'ate_per_variable_default_std', 'diff_ate_per_serie_variable_std', 'diff_ate_per_variable_std']
    if num_seed_i > 100:
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_all_"+model_name+"_mean.pkl", "wb") as datei:
                pickle.dump((cb_per_serie_variable_mean, cb_per_variable_mean, ate_per_serie_variable_mean, ate_per_variable_mean, ate_per_serie_variable_default, ate_per_variable_default, diff_ate_per_serie_variable_mean, diff_ate_per_variable_mean,list_mean),datei)        
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_all_"+model_name+"_std.pkl", "wb") as datei:
                pickle.dump((cb_per_serie_variable_std, cb_per_variable_std, ate_per_serie_variable_std, ate_per_variable_std,0,0, diff_ate_per_serie_variable_std, diff_ate_per_variable_std,list_std),datei)        
    else:
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_all_"+model_name+"_mean1.pkl", "wb") as datei:
                pickle.dump((cb_per_serie_variable_mean, cb_per_variable_mean, ate_per_serie_variable_mean, ate_per_variable_mean, ate_per_serie_variable_default, ate_per_variable_default, diff_ate_per_serie_variable_mean, diff_ate_per_variable_mean,list_mean),datei)        
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(cf)+"_seed_all_"+model_name+"_std1.pkl", "wb") as datei:
                pickle.dump((cb_per_serie_variable_std, cb_per_variable_std, ate_per_serie_variable_std, ate_per_variable_std,0,0, diff_ate_per_serie_variable_std, diff_ate_per_variable_std,list_std),datei)        
   
# %% 
if num_seed_i > 100:
        #1        
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(1)+"_seed_all_"+model_name+"_mean.pkl", "rb") as datei:
                cb_per_serie_variable_mean_1, cb_per_variable_mean_1, ate_per_serie_variable_mean_1, ate_per_variable_mean_1, ate_per_serie_variable_default_1, ate_per_variable_default_1, diff_ate_per_serie_variable_mean_1, diff_ate_per_variable_mean_1,list_mean_1 =  pickle.load(datei)

        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(1)+"_seed_all_"+model_name+"_std.pkl", "rb") as datei:
                cb_per_serie_variable_std_1, cb_per_variable_std_1, ate_per_serie_variable_std_1, ate_per_variable_std_1, ate_per_serie_variable_default_std_1, ate_per_variable_default_std_1, diff_ate_per_serie_variable_std_1, diff_ate_per_variable_std_1,list_std_1 =  pickle.load(datei)

        #2
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(2)+"_seed_all_"+model_name+"_mean.pkl", "rb") as datei:
                cb_per_serie_variable_mean_2, cb_per_variable_mean_2, ate_per_serie_variable_mean_2, ate_per_variable_mean_2, ate_per_serie_variable_default_2, ate_per_variable_default_2, diff_ate_per_serie_variable_mean_2, diff_ate_per_variable_mean_2,list_mean_2 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(2)+"_seed_all_"+model_name+"_std.pkl", "rb") as datei:
                cb_per_serie_variable_std_2, cb_per_variable_std_2, ate_per_serie_variable_std_2, ate_per_variable_std_2, ate_per_serie_variable_default_std_2, ate_per_variable_default_std_2, diff_ate_per_serie_variable_std_2, diff_ate_per_variable_std_2,list_std_2 =  pickle.load(datei)

        #3
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(3)+"_seed_all_"+model_name+"_mean.pkl", "rb") as datei:
                cb_per_serie_variable_mean_3, cb_per_variable_mean_3, ate_per_serie_variable_mean_3, ate_per_variable_mean_3, ate_per_serie_variable_default_3, ate_per_variable_default_3, diff_ate_per_serie_variable_mean_3, diff_ate_per_variable_mean_3,list_mean_3 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(3)+"_seed_all_"+model_name+"_std.pkl", "rb") as datei:
                cb_per_serie_variable_std_3, cb_per_variable_std_3, ate_per_serie_variable_std_3, ate_per_variable_std_3, ate_per_serie_variable_default_std_3, ate_per_variable_default_std_3, diff_ate_per_serie_variable_std_3, diff_ate_per_variable_std_3,list_std_3 =  pickle.load(datei)

        #4
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(4)+"_seed_all_"+model_name+"_mean.pkl", "rb") as datei:
                cb_per_serie_variable_mean_4, cb_per_variable_mean_4, ate_per_serie_variable_mean_4, ate_per_variable_mean_4, ate_per_serie_variable_default_4, ate_per_variable_default_4, diff_ate_per_serie_variable_mean_4, diff_ate_per_variable_mean_4,list_mean_4 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(4)+"_seed_all_"+model_name+"_std.pkl", "rb") as datei:
                cb_per_serie_variable_std_4, cb_per_variable_std_4, ate_per_serie_variable_std_4, ate_per_variable_std_4, ate_per_serie_variable_default_std_4, ate_per_variable_default_std_4, diff_ate_per_serie_variable_std_4, diff_ate_per_variable_std_4,list_std_4 =  pickle.load(datei)
        #5
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(5)+"_seed_all_"+model_name+"_mean.pkl", "rb") as datei:
                cb_per_serie_variable_mean_5, cb_per_variable_mean_5, ate_per_serie_variable_mean_5, ate_per_variable_mean_5, ate_per_serie_variable_default_5, ate_per_variable_default_5, diff_ate_per_serie_variable_mean_5, diff_ate_per_variable_mean_5,list_mean_5 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(5)+"_seed_all_"+model_name+"_std.pkl", "rb") as datei:
                cb_per_serie_variable_std_5, cb_per_variable_std_5, ate_per_serie_variable_std_5, ate_per_variable_std_5, ate_per_serie_variable_default_std_5, ate_per_variable_default_std_5, diff_ate_per_serie_variable_std_5, diff_ate_per_variable_std_5,list_std_5 =  pickle.load(datei)
else:
        #1        
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(1)+"_seed_all_"+model_name+"_mean1.pkl", "rb") as datei:
                cb_per_serie_variable_mean_1, cb_per_variable_mean_1, ate_per_serie_variable_mean_1, ate_per_variable_mean_1, ate_per_serie_variable_default_1, ate_per_variable_default_1, diff_ate_per_serie_variable_mean_1, diff_ate_per_variable_mean_1,list_mean_1 =  pickle.load(datei)

        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(1)+"_seed_all_"+model_name+"_std1.pkl", "rb") as datei:
                cb_per_serie_variable_std_1, cb_per_variable_std_1, ate_per_serie_variable_std_1, ate_per_variable_std_1, ate_per_serie_variable_default_std_1, ate_per_variable_default_std_1, diff_ate_per_serie_variable_std_1, diff_ate_per_variable_std_1,list_std_1 =  pickle.load(datei)

        #2
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(2)+"_seed_all_"+model_name+"_mean1.pkl", "rb") as datei:
                cb_per_serie_variable_mean_2, cb_per_variable_mean_2, ate_per_serie_variable_mean_2, ate_per_variable_mean_2, ate_per_serie_variable_default_2, ate_per_variable_default_2, diff_ate_per_serie_variable_mean_2, diff_ate_per_variable_mean_2,list_mean_2 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(2)+"_seed_all_"+model_name+"_std1.pkl", "rb") as datei:
                cb_per_serie_variable_std_2, cb_per_variable_std_2, ate_per_serie_variable_std_2, ate_per_variable_std_2, ate_per_serie_variable_default_std_2, ate_per_variable_default_std_2, diff_ate_per_serie_variable_std_2, diff_ate_per_variable_std_2,list_std_2 =  pickle.load(datei)

        #3
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(3)+"_seed_all_"+model_name+"_mean1.pkl", "rb") as datei:
                cb_per_serie_variable_mean_3, cb_per_variable_mean_3, ate_per_serie_variable_mean_3, ate_per_variable_mean_3, ate_per_serie_variable_default_3, ate_per_variable_default_3, diff_ate_per_serie_variable_mean_3, diff_ate_per_variable_mean_3,list_mean_3 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(3)+"_seed_all_"+model_name+"_std1.pkl", "rb") as datei:
                cb_per_serie_variable_std_3, cb_per_variable_std_3, ate_per_serie_variable_std_3, ate_per_variable_std_3, ate_per_serie_variable_default_std_3, ate_per_variable_default_std_3, diff_ate_per_serie_variable_std_3, diff_ate_per_variable_std_3,list_std_3 =  pickle.load(datei)

        #4
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(4)+"_seed_all_"+model_name+"_mean1.pkl", "rb") as datei:
                cb_per_serie_variable_mean_4, cb_per_variable_mean_4, ate_per_serie_variable_mean_4, ate_per_variable_mean_4, ate_per_serie_variable_default_4, ate_per_variable_default_4, diff_ate_per_serie_variable_mean_4, diff_ate_per_variable_mean_4,list_mean_4 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(4)+"_seed_all_"+model_name+"_std1.pkl", "rb") as datei:
                cb_per_serie_variable_std_4, cb_per_variable_std_4, ate_per_serie_variable_std_4, ate_per_variable_std_4, ate_per_serie_variable_default_std_4, ate_per_variable_default_std_4, diff_ate_per_serie_variable_std_4, diff_ate_per_variable_std_4,list_std_4 =  pickle.load(datei)


        #5
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(5)+"_seed_all_"+model_name+"_mean1.pkl", "rb") as datei:
                cb_per_serie_variable_mean_5, cb_per_variable_mean_5, ate_per_serie_variable_mean_5, ate_per_variable_mean_5, ate_per_serie_variable_default_5, ate_per_variable_default_5, diff_ate_per_serie_variable_mean_5, diff_ate_per_variable_mean_5,list_mean_5 =  pickle.load(datei)
        with open(save_path+"data_pred_cb_ate_semisynMimicIV_CF_"+str(5)+"_seed_all_"+model_name+"_std1.pkl", "rb") as datei:
                cb_per_serie_variable_std_5, cb_per_variable_std_5, ate_per_serie_variable_std_5, ate_per_variable_std_5, ate_per_serie_variable_default_std_5, ate_per_variable_default_std_5, diff_ate_per_serie_variable_std_5, diff_ate_per_variable_std_5,list_std_5 =  pickle.load(datei)


# %%
num = 24
### CB ###
cb_per_variable_mean_all = [cb_per_variable_mean_1, cb_per_variable_mean_2, cb_per_variable_mean_3, cb_per_variable_mean_4, cb_per_variable_mean_5]
cb_per_variable_std_all = [cb_per_variable_std_1, cb_per_variable_std_2, cb_per_variable_std_3, cb_per_variable_std_4, cb_per_variable_std_5]

cb_per_variable_mean_num = [cb_per_serie_variable_mean_1[:num].mean(dim=0), cb_per_serie_variable_mean_2[:num].mean(dim=0), cb_per_serie_variable_mean_3[:num].mean(dim=0), cb_per_serie_variable_mean_4[:num].mean(dim=0), cb_per_serie_variable_mean_5[:num].mean(dim=0)]
cb_per_variable_std_num = [cb_per_serie_variable_std_1[:num].mean(dim=0), cb_per_serie_variable_std_2[:num].mean(dim=0), cb_per_serie_variable_std_3[:num].mean(dim=0), cb_per_serie_variable_std_4[:num].mean(dim=0), cb_per_serie_variable_std_5[:num].mean(dim=0)]
### ATE ###
ate_per_variable_mean_all = [ate_per_variable_mean_1, ate_per_variable_mean_2, ate_per_variable_mean_3, ate_per_variable_mean_4, ate_per_variable_mean_5]
ate_per_variable_std_all = [ate_per_variable_std_1, ate_per_variable_std_2, ate_per_variable_std_3, ate_per_variable_std_4, ate_per_variable_std_5]
ate_per_variable_mean_num = [ate_per_serie_variable_mean_1[:num].mean(dim=0), ate_per_serie_variable_mean_2[:num].mean(dim=0), ate_per_serie_variable_mean_3[:num].mean(dim=0), ate_per_serie_variable_mean_4[:num].mean(dim=0), ate_per_serie_variable_mean_5[:num].mean(dim=0)]
ate_per_variable_std_num = [ate_per_serie_variable_std_1[:num].mean(dim=0), ate_per_serie_variable_std_2[:num].mean(dim=0), ate_per_serie_variable_std_3[:num].mean(dim=0), ate_per_serie_variable_std_4[:num].mean(dim=0), ate_per_serie_variable_std_5[:num].mean(dim=0)]


### ATE diff ###
diff_ate_per_variable_mean_all = [diff_ate_per_variable_mean_1, diff_ate_per_variable_mean_2, diff_ate_per_variable_mean_3, diff_ate_per_variable_mean_4, diff_ate_per_variable_mean_5]
diff_ate_per_variable_std_all = [diff_ate_per_variable_std_1, diff_ate_per_variable_std_2, diff_ate_per_variable_std_3, diff_ate_per_variable_std_4, diff_ate_per_variable_std_5]

diff_ate_per_variable_mean_num = [diff_ate_per_serie_variable_mean_1[:num].mean(dim=0), diff_ate_per_serie_variable_mean_2[:num].mean(dim=0), diff_ate_per_serie_variable_mean_3[:num].mean(dim=0), diff_ate_per_serie_variable_mean_4[:num].mean(dim=0), diff_ate_per_serie_variable_mean_5[:num].mean(dim=0)]
diff_ate_per_variable_std_num = [diff_ate_per_serie_variable_std_1[:num].mean(dim=0), diff_ate_per_serie_variable_std_2[:num].mean(dim=0), diff_ate_per_serie_variable_std_3[:num].mean(dim=0), diff_ate_per_serie_variable_std_4[:num].mean(dim=0), diff_ate_per_serie_variable_std_5[:num].mean(dim=0)]

### ATE default ###
ate_per_variable_default_all = [ate_per_variable_default_1, ate_per_variable_default_2, ate_per_variable_default_3, ate_per_variable_default_4, ate_per_variable_default_5]

ate_per_variable_default_all_num = [ate_per_serie_variable_default_1[:num].mean(dim=0), ate_per_serie_variable_default_2[:num].mean(dim=0), ate_per_serie_variable_default_3[:num].mean(dim=0), ate_per_serie_variable_default_4[:num].mean(dim=0), ate_per_serie_variable_default_5[:num].mean(dim=0)]


#ATE per variable over time 
ate_default_num = ate_per_variable_default_all_num#ate_per_serie_variable_default_1[:num]
ate_over_num_time_mean = torch.stack(ate_per_variable_mean_num)#CF 1 to 5
ate_over_num_time_std = torch.stack(ate_per_variable_std_num)#CF 1 to 5
ate_over_num_time_mean_m = ate_over_num_time_mean.mean(dim=0)
ate_over_num_time_std_m = ate_over_num_time_std.mean(dim=0)

#ATE diff mean over num time, from CF 1 to 5, compared to default CF6
#diff_ate_over_num_time_mean = torch.stack(diff_ate_per_variable_mean_num[:-1])#CF 1 to 5
#diff_ate_over_num_time_std = torch.stack(diff_ate_per_variable_std_num[:-1])#CF 1 to 5
#diff_ate_over_num_time_mean_m = diff_ate_over_num_time_mean.mean(dim=0)
#diff_ate_over_num_time_std_m = diff_ate_over_num_time_std.mean(dim=0)

#CB mean over num time, from CF 1 to 5, compared to default CF6, abs(CB) sollte gleich sein
cb_over_num_time_mean = torch.stack(cb_per_variable_mean_num)
cb_over_num_time_std = torch.stack(cb_per_variable_std_num)
cb_over_num_time_mean_m = cb_over_num_time_mean.mean(dim=0)
cb_cb_over_num_time_std_m = cb_over_num_time_std.mean(dim=0)


#ATE per variable over time 
ate_default = ate_per_variable_default_all#ate_per_serie_variable_default_1
ate_over_time_mean = torch.stack(ate_per_variable_mean_all)
ate_over_time_std = torch.stack(ate_per_variable_std_all)
ate_over_time_mean_m = ate_over_time_mean.mean(dim=0)
ate_over_time_std_m = ate_over_time_std.mean(dim=0)

#ATE diff mean over num time, from CF 1 to 5,
#diff_ate_over_time_mean = torch.stack(diff_ate_per_variable_mean_all[:-1])#CF 1 to 5
#diff_ate_over_time_std = torch.stack(diff_ate_per_variable_std_all[:-1])#CF 1 to 5
#diff_ate_over_time_mean_m = diff_ate_over_time_mean.mean(dim=0)
#diff_ate_over_time_std_m = diff_ate_over_time_std.mean(dim=0)

#CB mean over num time, from CF 1 to 5, compared to default CF6, abs(CB) sollte gleich sein
cb_over_time_mean = torch.stack(cb_per_variable_mean_all)
cb_over_time_std = torch.stack(cb_per_variable_std_all)
cb_over_time_mean_m = cb_over_time_mean.mean(dim=0)
cb_cb_over_time_std_m = cb_over_time_std.mean(dim=0)

cb_over_num_time_mean_m = torch.abs(cb_over_num_time_mean).mean(dim=0)
cb_cb_over_num_time_std_m = torch.abs(cb_over_num_time_std).mean(dim=0)
cb_over_num_time_mean_m 
