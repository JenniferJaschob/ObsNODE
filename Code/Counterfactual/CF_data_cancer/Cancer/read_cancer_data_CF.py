#%%
import os
import numpy as np
import torch    
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Counterfactual_Datasets_For_ATE/Cancer/src/utils/')
#os.chdir('/Users/jaschob/Schreibtisch/Paper1_cc/Counterfactual_Datasets_For_ATE/Cancer/src/utils/')
from data_utils import read_from_file, process_data, read_from_file

import pickle

torch.manual_seed(1)
np.random.seed(1)

gamma = 4

for i in range(1,6):
    cf = i
    test_data = read_from_file('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Counterfactual_Datasets_For_ATE/results/dataset_'+str(cf)+'.p')

    x_test = torch.cat([torch.tensor(test_data['cancer_volume']).unsqueeze(2),torch.tensor(test_data['toxicity']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)
    u_test = torch.cat([torch.tensor(test_data['chemo_dosage']).unsqueeze(2),torch.tensor(test_data['radio_dosage']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)
    u_test2 = torch.cat([torch.tensor(test_data['chemo_application']).unsqueeze(2),torch.tensor(test_data['radio_application']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)

    #define time values
    dt_test, dp_test, dv_test = x_test.size()
    t_test_1 = np.linspace(0, dt_test-1, dt_test, dtype=np.float32)  
    t_test = torch.tensor(np.tile(t_test_1[:, np.newaxis, np.newaxis], (1, dp_test, dv_test)), dtype=torch.float)

    value = 0
    y_test =  x_test[...,value]
    if cf == 5:
        u_test[:,:,0]  = u_test2[:,:,0] 
    elif cf== 3:
        u_test[:,:,0]  = u_test2[:,:,0]   

    save_path = '/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Counterfactual_Datasets_For_ATE/results/' 
    with open(save_path+"data_test_cancer_gamma_"+str(gamma)+"_CF_"+str(cf)+".pkl", "wb") as datei:
        pickle.dump((t_test, x_test,y_test), datei)     

    with open(save_path+"data_test_cancer_value_u_gamma_"+str(gamma)+"_CF_"+str(cf)+".pkl", "wb") as datei:
        pickle.dump(u_test,datei)
    
  
    
#