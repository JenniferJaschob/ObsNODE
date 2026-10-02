#%%
import torch
import numpy as np
import pickle

import os
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Model_observableNODE_paper')
from utils_NODE import  LSTMRecognitionModel_with_nan, MyNeuralODE_withNNs, MyModel_adjoint, predict
from utils_NODE import *

num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)
#%%
####
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#read Data

for i in range(1, 7):
    cf = i
    path_data = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Counterfactual_Datasets_For_ATE/results/'

    with open(path_data+'data_test_cancer_gamma_4_CF_'+str(6)+'.pkl', "rb") as datei:
        data_test_default = pickle.load(datei)
                        
    with open(path_data+'data_test_cancer_value_u_gamma_4_CF_'+str(6)+'.pkl', "rb") as datei:
        u_test_default = pickle.load(datei)

    with open(path_data+'data_test_cancer_gamma_4_CF_'+str(cf)+'.pkl', "rb") as datei:
        data_test = pickle.load(datei)
                        
    with open(path_data+'data_test_cancer_value_u_gamma_4_CF_'+str(cf)+'.pkl', "rb") as datei:
        u_test = pickle.load(datei)

    path_data2 = '/Users/jaschob/Desktop/data_cancer/'


    with open(path_data2+"data_train_cancer_DoseAI.pkl", "rb") as datei:
        data_train = pickle.load(datei) 

    with open(path_data2+"data_train_cancer_value_u_DoseAI.pkl", "rb") as datei:
        u_train = pickle.load(datei)       
                    
    t_test, x_test, _    =  data_test
    u_test[torch.isnan(u_test)] = 0

    t_test_default, x_test_default, _    =  data_test_default
    u_test_default[torch.isnan(u_test_default)] = 0


    t_train, x_train, _    =  data_train
    u_train[torch.isnan(u_train)] = 0


    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True)

    # define Parameters
    #hyperopt NODE
    params_dic ={'n_z': 2,
                'batch_size': 96,
                'learning_rate': 0.001,
                'hidden_dim_obs': 256,
                'hidden_dim_node': 64,
                'num_layers_node': 8,
                'activation_node': 'sigmoid'}


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


    # define Model ################################################################
    myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
      
    myNODE = MyNeuralODE_withNNs( activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
    model = MyModel_adjoint(myNODE, myObserver).to(device)

    ####
    read_path = '/Users/jaschob/Desktop/Out_NODE/model_state_dict_seed_'+str(num_seed_i)+'_cancer.pth'
    model.load_state_dict(torch.load(read_path,map_location=torch.device('cpu')))


    # ##############################################################################
    # Prediction
    #train test split
    num_time = 24
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
    save_path =  '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Pred_CF_cancer/NODE/' 
    with open(save_path+"data_pred_cancer_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_NODE.pkl", "wb") as datei:
            pickle.dump(x_pred_i,datei)


    x_pred_i = (x_pred_i - mean) / std
    x_test_after = (x_test_after - mean) / std
    x_test_after_default = (x_test_after_default - mean) / std

    diff_predcf_truecf = x_pred_i - x_test_after
    cb_per_serie_variable = diff_predcf_truecf.mean(dim=1)
    cb_per_variable = diff_predcf_truecf.mean(dim=(0, 1))

    diff_predcf_true = x_pred_i - x_test_after_default
    ate_per_serie_variable = diff_predcf_true.mean(dim=1)
    ate_per_variable = diff_predcf_true.mean(dim=(0, 1))

    diff_cf_true = x_test_after - x_test_after_default
    ate_per_serie_variable_default = diff_cf_true.mean(dim=1)
    ate_per_variable_default = diff_cf_true.mean(dim=(0, 1))

    list = ['cb_per_serie_variable', 'cb_per_variable', 'ate_per_serie_variable', 'ate_per_variable', 'ate_per_serie_variable_default', 'ate_per_variable_default']
    with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_NODE.pkl", "wb") as datei:
        pickle.dump((cb_per_serie_variable, cb_per_variable, ate_per_serie_variable, ate_per_variable, ate_per_serie_variable_default, ate_per_variable_default),datei)        


# %%


for cf in range(1, 6):
    
    cb_per_variable_list, cb_per_serie_variable_list = [], []
    ate_per_variable_list, ate_per_serie_variable_list  = [], []

    for num_seed_i in range(1, 6):
        save_path = save_path = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Pred_CF_cancer/NODE/' 
        with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(cf)+"_seed_"+str(num_seed_i)+"_NODE.pkl", "rb") as datei:
            cb_per_serie_variable, cb_per_variable, ate_per_serie_variable, ate_per_variable, ate_per_serie_variable_default, ate_per_variable_default = pickle.load(datei)

        cb_per_variable_list.append(cb_per_variable)
        cb_per_serie_variable_list.append(cb_per_serie_variable)
        ate_per_variable_list.append(ate_per_variable)
        ate_per_serie_variable_list.append(ate_per_serie_variable)

    cb_per_variable_mean = torch.stack(cb_per_variable_list).mean(dim=0)
    cb_per_serie_variable_mean = torch.stack(cb_per_serie_variable_list).mean(dim=0)
    ate_per_variable_mean = torch.stack(ate_per_variable_list).mean(dim=0)
    ate_per_serie_variable_mean = torch.stack(ate_per_serie_variable_list).mean(dim=0)    

    cb_per_variable_std = torch.stack(cb_per_variable_list).std(dim=0)
    cb_per_serie_variable_std = torch.stack(cb_per_serie_variable_list).std(dim=0)
    ate_per_variable_std = torch.stack(ate_per_variable_list).std(dim=0)
    ate_per_serie_variable_std = torch.stack(ate_per_serie_variable_list).std(dim=0)    


    list_mean = ['cb_per_serie_variable_mean', 'cb_per_variable_mean', 'ate_per_serie_variable_mean', 'ate_per_variable_mean', 'ate_per_serie_variable_default_mean', 'ate_per_variable_default_mean']
    list_std = ['cb_per_serie_variable_std', 'cb_per_variable_std', 'ate_per_serie_variable_std', 'ate_per_variable_std', 'ate_per_serie_variable_default_std', 'ate_per_variable_default_std']

    with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(cf)+"_seed_all_NODE_mean.pkl", "wb") as datei:
            pickle.dump((cb_per_serie_variable_mean, cb_per_variable_mean, ate_per_serie_variable_mean, ate_per_variable_mean, ate_per_serie_variable_default, ate_per_variable_default),datei)        
    with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(cf)+"_seed_all_NODE_std.pkl", "wb") as datei:
            pickle.dump((cb_per_serie_variable_std, cb_per_variable_std, ate_per_serie_variable_std, ate_per_variable_std,0,0),datei)        

# %%
save_path = save_path = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Pred_CF_cancer/NODE/' 
        
#1        
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(1)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_1, cb_per_variable_mean_1, ate_per_serie_variable_mean_1, ate_per_variable_mean_1, ate_per_serie_variable_default_1, ate_per_variable_default_1 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(1)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_1, cb_per_variable_std_1, ate_per_serie_variable_std_1, ate_per_variable_std_1, ate_per_serie_variable_default_std_1, ate_per_variable_default_std_1 =  pickle.load(datei)

#2
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(2)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_2, cb_per_variable_mean_2, ate_per_serie_variable_mean_2, ate_per_variable_mean_2, ate_per_serie_variable_default_2, ate_per_variable_default_2 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(2)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_2, cb_per_variable_std_2, ate_per_serie_variable_std_2, ate_per_variable_std_2, ate_per_serie_variable_default_std_2, ate_per_variable_default_std_2 =  pickle.load(datei)

#3
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(3)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_3, cb_per_variable_mean_3, ate_per_serie_variable_mean_3, ate_per_variable_mean_3, ate_per_serie_variable_default_3, ate_per_variable_default_3 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(3)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_3, cb_per_variable_std_3, ate_per_serie_variable_std_3, ate_per_variable_std_3, ate_per_serie_variable_default_std_3, ate_per_variable_default_std_3 =  pickle.load(datei)

#4
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(4)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_4, cb_per_variable_mean_4, ate_per_serie_variable_mean_4, ate_per_variable_mean_4, ate_per_serie_variable_default_4, ate_per_variable_default_4 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(4)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_4, cb_per_variable_std_4, ate_per_serie_variable_std_4, ate_per_variable_std_4, ate_per_serie_variable_default_std_4, ate_per_variable_default_std_4 =  pickle.load(datei)


#5
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(5)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_5, cb_per_variable_mean_5, ate_per_serie_variable_mean_5, ate_per_variable_mean_5, ate_per_serie_variable_default_5, ate_per_variable_default_5 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(5)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_5, cb_per_variable_std_5, ate_per_serie_variable_std_5, ate_per_variable_std_5, ate_per_serie_variable_default_std_5, ate_per_variable_default_std_5 =  pickle.load(datei)


#6 default
with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(6)+"_seed_all_NODE_mean.pkl", "rb") as datei:
        cb_per_serie_variable_mean_6, cb_per_variable_mean_6, ate_per_serie_variable_mean_6, ate_per_variable_mean_6, ate_per_serie_variable_default_6, ate_per_variable_default_6 =  pickle.load(datei)

with open(save_path+"data_pred_cb_ate_cancer_CF_"+str(6)+"_seed_all_NODE_std.pkl", "rb") as datei:
        cb_per_serie_variable_std_6, cb_per_variable_std_6, ate_per_serie_variable_std_6, ate_per_variable_std_6, ate_per_serie_variable_default_std_6, ate_per_variable_default_std_6 =  pickle.load(datei)

# %%
cb_per_variable_mean_all = [cb_per_variable_mean_1, cb_per_variable_mean_2, cb_per_variable_mean_3, cb_per_variable_mean_4, cb_per_variable_mean_5, cb_per_variable_mean_6]
cb_per_variable_std_all = [cb_per_variable_std_1, cb_per_variable_std_2, cb_per_variable_std_3, cb_per_variable_std_4, cb_per_variable_std_5, cb_per_variable_std_6]


cb_per_variable_mean_6 = [cb_per_serie_variable_mean_1[:6].mean(dim=0), cb_per_serie_variable_mean_2[:6].mean(dim=0), cb_per_serie_variable_mean_3[:6].mean(dim=0), cb_per_serie_variable_mean_4[:6].mean(dim=0), cb_per_serie_variable_mean_5[:6].mean(dim=0), cb_per_serie_variable_mean_6[:6].mean(dim=0)]
cb_per_variable_mean_5 = [cb_per_serie_variable_mean_1[:5].mean(dim=0), cb_per_serie_variable_mean_2[:5].mean(dim=0), cb_per_serie_variable_mean_3[:5].mean(dim=0), cb_per_serie_variable_mean_4[:5].mean(dim=0), cb_per_serie_variable_mean_5[:5].mean(dim=0), cb_per_serie_variable_mean_6[:5].mean(dim=0)]
cb_per_variable_mean_4 = [cb_per_serie_variable_mean_1[:4].mean(dim=0), cb_per_serie_variable_mean_2[:4].mean(dim=0), cb_per_serie_variable_mean_3[:4].mean(dim=0), cb_per_serie_variable_mean_4[:4].mean(dim=0), cb_per_serie_variable_mean_5[:4].mean(dim=0), cb_per_serie_variable_mean_6[:4].mean(dim=0)]
cb_per_variable_mean_3 = [cb_per_serie_variable_mean_1[:3].mean(dim=0), cb_per_serie_variable_mean_2[:3].mean(dim=0), cb_per_serie_variable_mean_3[:3].mean(dim=0), cb_per_serie_variable_mean_4[:3].mean(dim=0), cb_per_serie_variable_mean_5[:3].mean(dim=0), cb_per_serie_variable_mean_6[:3].mean(dim=0)]
cb_per_variable_mean_2 = [cb_per_serie_variable_mean_1[:2].mean(dim=0), cb_per_serie_variable_mean_2[:2].mean(dim=0), cb_per_serie_variable_mean_3[:2].mean(dim=0), cb_per_serie_variable_mean_4[:2].mean(dim=0), cb_per_serie_variable_mean_5[:2].mean(dim=0), cb_per_serie_variable_mean_6[:2].mean(dim=0)]
cb_per_variable_mean_1 = [cb_per_serie_variable_mean_1[:1].mean(dim=0), cb_per_serie_variable_mean_2[:1].mean(dim=0), cb_per_serie_variable_mean_3[:1].mean(dim=0), cb_per_serie_variable_mean_4[:1].mean(dim=0), cb_per_serie_variable_mean_5[:1].mean(dim=0), cb_per_serie_variable_mean_6[:1].mean(dim=0)]

cb_per_variable_std_6 = [cb_per_serie_variable_std_1[:6].mean(dim=0), cb_per_serie_variable_std_2[:6].mean(dim=0), cb_per_serie_variable_std_3[:6].mean(dim=0), cb_per_serie_variable_std_4[:6].mean(dim=0), cb_per_serie_variable_std_5[:6].mean(dim=0), cb_per_serie_variable_std_6[:6].mean(dim=0)]
cb_per_variable_std_5 = [cb_per_serie_variable_std_1[:5].mean(dim=0), cb_per_serie_variable_std_2[:5].mean(dim=0), cb_per_serie_variable_std_3[:5].mean(dim=0), cb_per_serie_variable_std_4[:5].mean(dim=0), cb_per_serie_variable_std_5[:5].mean(dim=0), cb_per_serie_variable_std_6[:5].mean(dim=0)]
cb_per_variable_std_4 = [cb_per_serie_variable_std_1[:4].mean(dim=0), cb_per_serie_variable_std_2[:4].mean(dim=0), cb_per_serie_variable_std_3[:4].mean(dim=0), cb_per_serie_variable_std_4[:4].mean(dim=0), cb_per_serie_variable_std_5[:4].mean(dim=0), cb_per_serie_variable_std_6[:4].mean(dim=0)]
cb_per_variable_std_3 = [cb_per_serie_variable_std_1[:3].mean(dim=0), cb_per_serie_variable_std_2[:3].mean(dim=0), cb_per_serie_variable_std_3[:3].mean(dim=0), cb_per_serie_variable_std_4[:3].mean(dim=0), cb_per_serie_variable_std_5[:3].mean(dim=0), cb_per_serie_variable_std_6[:3].mean(dim=0)]
cb_per_variable_std_2 = [cb_per_serie_variable_std_1[:2].mean(dim=0), cb_per_serie_variable_std_2[:2].mean(dim=0), cb_per_serie_variable_std_3[:2].mean(dim=0), cb_per_serie_variable_std_4[:2].mean(dim=0), cb_per_serie_variable_std_5[:2].mean(dim=0), cb_per_serie_variable_std_6[:2].mean(dim=0)]
cb_per_variable_std_1 = [cb_per_serie_variable_std_1[:1].mean(dim=0), cb_per_serie_variable_std_2[:1].mean(dim=0), cb_per_serie_variable_std_3[:1].mean(dim=0), cb_per_serie_variable_std_4[:1].mean(dim=0), cb_per_serie_variable_std_5[:1].mean(dim=0), cb_per_serie_variable_std_6[:1].mean(dim=0)]

#%%
ate_per_variable_mean_all = [ate_per_variable_mean_1, ate_per_variable_mean_2, ate_per_variable_mean_3, ate_per_variable_mean_4, ate_per_variable_mean_5, ate_per_variable_mean_6]
ate_per_variable_std_all = [ate_per_variable_std_1, ate_per_variable_std_2, ate_per_variable_std_3, ate_per_variable_std_4, ate_per_variable_std_5, ate_per_variable_std_6]

ate_per_variable_mean_1_abs_diff = torch.abs(ate_per_variable_mean_1 - ate_per_variable_default_1)
ate_per_variable_mean_2_abs_diff = torch.abs(ate_per_variable_mean_2 - ate_per_variable_default_1)
ate_per_variable_mean_3_abs_diff = torch.abs(ate_per_variable_mean_3 - ate_per_variable_default_1)
ate_per_variable_mean_4_abs_diff = torch.abs(ate_per_variable_mean_4 - ate_per_variable_default_1)
ate_per_variable_mean_5_abs_diff = torch.abs(ate_per_variable_mean_5 - ate_per_variable_default_1)
ate_per_variable_mean_6_abs_diff = torch.abs(ate_per_variable_mean_6 - ate_per_variable_default_1)
ate_per_variable_mean_all_abs_diff = [ate_per_variable_mean_1_abs_diff, ate_per_variable_mean_2_abs_diff, ate_per_variable_mean_3_abs_diff, ate_per_variable_mean_4_abs_diff, ate_per_variable_mean_5_abs_diff, ate_per_variable_mean_6_abs_diff]


# %%

ate_per_variable_mean_1_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_1[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_2_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_2[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_3_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_3[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_4_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_4[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_5_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_5[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_6_abs_diff_6 = torch.abs(ate_per_serie_variable_mean_6[:6].mean(dim=0) - ate_per_serie_variable_default_1[:6].mean(dim=0))
ate_per_variable_mean_all_abs_diff_6 = [ate_per_variable_mean_1_abs_diff_6, ate_per_variable_mean_2_abs_diff_6, ate_per_variable_mean_3_abs_diff_6, ate_per_variable_mean_4_abs_diff_6, ate_per_variable_mean_5_abs_diff_6, ate_per_variable_mean_6_abs_diff_6]



ate_per_variable_mean_1_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_1[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_2_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_2[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_3_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_3[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_4_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_4[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_5_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_5[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_6_abs_diff_5 = torch.abs(ate_per_serie_variable_mean_6[:5].mean(dim=0) - ate_per_serie_variable_default_1[:5].mean(dim=0))
ate_per_variable_mean_all_abs_diff_5 = [ate_per_variable_mean_1_abs_diff_5, ate_per_variable_mean_2_abs_diff_5, ate_per_variable_mean_3_abs_diff_5, ate_per_variable_mean_4_abs_diff_5, ate_per_variable_mean_5_abs_diff_5, ate_per_variable_mean_6_abs_diff_5]


ate_per_variable_mean_1_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_1[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_2_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_2[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_3_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_3[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_4_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_4[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_5_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_5[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_6_abs_diff_4 = torch.abs(ate_per_serie_variable_mean_6[:4].mean(dim=0) - ate_per_serie_variable_default_1[:4].mean(dim=0))
ate_per_variable_mean_all_abs_diff_4 = [ate_per_variable_mean_1_abs_diff_4, ate_per_variable_mean_2_abs_diff_4, ate_per_variable_mean_3_abs_diff_4, ate_per_variable_mean_4_abs_diff_4, ate_per_variable_mean_5_abs_diff_4, ate_per_variable_mean_6_abs_diff_4]

ate_per_variable_mean_1_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_1[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_2_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_2[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_3_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_3[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_4_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_4[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_5_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_5[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_6_abs_diff_3 = torch.abs(ate_per_serie_variable_mean_6[:3].mean(dim=0) - ate_per_serie_variable_default_1[:3].mean(dim=0))
ate_per_variable_mean_all_abs_diff_3 = [ate_per_variable_mean_1_abs_diff_3, ate_per_variable_mean_2_abs_diff_3, ate_per_variable_mean_3_abs_diff_3, ate_per_variable_mean_4_abs_diff_3, ate_per_variable_mean_5_abs_diff_3, ate_per_variable_mean_6_abs_diff_3]

ate_per_variable_mean_1_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_1[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_2_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_2[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_3_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_3[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_4_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_4[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_5_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_5[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_6_abs_diff_2 = torch.abs(ate_per_serie_variable_mean_6[:2].mean(dim=0) - ate_per_serie_variable_default_1[:2].mean(dim=0))
ate_per_variable_mean_all_abs_diff_2 = [ate_per_variable_mean_1_abs_diff_2, ate_per_variable_mean_2_abs_diff_2, ate_per_variable_mean_3_abs_diff_2, ate_per_variable_mean_4_abs_diff_2, ate_per_variable_mean_5_abs_diff_2, ate_per_variable_mean_6_abs_diff_2]

ate_per_variable_mean_1_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_1[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_2_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_2[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_3_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_3[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_4_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_4[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_5_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_5[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_6_abs_diff_1 = torch.abs(ate_per_serie_variable_mean_6[:1].mean(dim=0) - ate_per_serie_variable_default_1[:1].mean(dim=0))
ate_per_variable_mean_all_abs_diff_1 = [ate_per_variable_mean_1_abs_diff_1, ate_per_variable_mean_2_abs_diff_1, ate_per_variable_mean_3_abs_diff_1, ate_per_variable_mean_4_abs_diff_1, ate_per_variable_mean_5_abs_diff_1, ate_per_variable_mean_6_abs_diff_1]
# %%
print('ATE 1:', ate_per_variable_mean_all_abs_diff_1 )
print('ATE 2:', ate_per_variable_mean_all_abs_diff_2)
print('ATE 3:', ate_per_variable_mean_all_abs_diff_3)
print('ATE 4:', ate_per_variable_mean_all_abs_diff_4)
print('ATE 5:', ate_per_variable_mean_all_abs_diff_5)
print('ATE 6:', ate_per_variable_mean_all_abs_diff_6)
print('ATE all:', ate_per_variable_mean_all_abs_diff)


# %%
#finale Auswahl horizen 1
ate_final_Cf_t0_Cf5_time1=torch.stack(ate_per_variable_mean_all_abs_diff_1[:-1])
cb_final_Cf_t0_Cf5_time1=torch.stack(cb_per_variable_mean_1[:-1])


#Exemplarisch über die Zeit wäre z.b 5 
cb_Cf5_time1_to_5 = cb_per_serie_variable_mean_5[:5]
cb_Cf5_time1_to_5_mean = cb_Cf5_time1_to_5.mean(dim=0)

ate_Cf5_time1_to_5 = torch.abs(ate_per_serie_variable_mean_5[:5]-ate_per_serie_variable_default_1[:5])
ate_Cf5_time1_to_5_mean = ate_Cf5_time1_to_5.mean(dim=0)

#%%

ate_per_variable_std_all_abs_diff_1 = [ate_per_serie_variable_std_1[:1].mean(dim=0) , ate_per_serie_variable_std_3[:1].mean(dim=0) , ate_per_serie_variable_std_4[:1].mean(dim=0) ,ate_per_serie_variable_std_5[:1].mean(dim=0)]


ate_final_Cf_t0_Cf5_time1_std = torch.stack(ate_per_variable_std_all_abs_diff_1[:-1])
cb_final_Cf_t0_Cf5_time1_std=torch.stack(cb_per_variable_std_2[:-1])


cb_Cf5_time1_to_5_std = cb_per_serie_variable_std_5[:5]
cb_Cf5_time1_to_5_mean_std = cb_Cf5_time1_to_5_std.mean(dim=0)

ate_Cf5_time1_to_5_std = ate_per_serie_variable_std_5[:5] 
ate_Cf5_time1_to_5_mean_std = ate_Cf5_time1_to_5_std.mean(dim=0)
# %%
ate_per_variable_mean_all_abs_diff_1 = [ate_per_serie_variable_mean_1[:1].mean(dim=0) , ate_per_serie_variable_mean_3[:1].mean(dim=0) , ate_per_serie_variable_mean_4[:1].mean(dim=0) ,ate_per_serie_variable_mean_5[:1].mean(dim=0)]
ate_per_variable_std_all_abs_diff_1 = [ate_per_serie_variable_std_1[:1].mean(dim=0) , ate_per_serie_variable_std_3[:1].mean(dim=0) , ate_per_serie_variable_std_4[:1].mean(dim=0) ,ate_per_serie_variable_std_5[:1].mean(dim=0)]

#default
ate_default = ate_per_serie_variable_default_1[:5]


ate_Cf5_time1_to_5_mean = ate_per_serie_variable_mean_5[:5] 
ate_Cf5_time1_to_5_mean_mean = ate_Cf5_time1_to_5_mean.mean(dim=0)
ate_Cf5_time1_to_5_std = ate_per_serie_variable_std_5[:5] 
ate_Cf5_time1_to_5_mean_std = ate_Cf5_time1_to_5_std.mean(dim=0)


ate_default_5 = ate_per_serie_variable_default_5
# %%
