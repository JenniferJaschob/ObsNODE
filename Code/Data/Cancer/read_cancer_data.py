import os
import numpy as np
import matplotlib.pyplot as plt
import torch    

os.chdir('/Users/jaschob/Desktop/DoseAI/src/utils/')
from data_utils import read_from_file, process_data, read_from_file

import pickle

torch.manual_seed(1)
np.random.seed(1)

gamma = 20

pickle_map = read_from_file('/Users/jaschob/Desktop/DoseAI/data/data_dict_'+str(gamma)+'_'+str(gamma)+'.p')


training_data = pickle_map['training_data']
x_train= torch.cat([torch.tensor(training_data['cancer_volume']).unsqueeze(2),torch.tensor(training_data['toxicity']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)
u_train = torch.cat([torch.tensor(training_data['chemo_dosage']).unsqueeze(2),torch.tensor(training_data['radio_dosage']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)

validation_data = pickle_map['validation_data']
x_validate = torch.cat([torch.tensor(validation_data['cancer_volume']).unsqueeze(2),torch.tensor(validation_data['toxicity']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)
u_validate = torch.cat([torch.tensor(validation_data['chemo_dosage']).unsqueeze(2),torch.tensor(validation_data['radio_dosage']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)

test_data = pickle_map['test_data_factuals']
x_test = torch.cat([torch.tensor(test_data['cancer_volume']).unsqueeze(2),torch.tensor(test_data['toxicity']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)
u_test = torch.cat([torch.tensor(test_data['chemo_dosage']).unsqueeze(2),torch.tensor(test_data['radio_dosage']).unsqueeze(2)], dim=2).transpose(0, 1).to(torch.float)

#define time values
dt_train, dp_train, dv_train = x_train.size()
t_train_1 = np.linspace(0, dt_train-1 , dt_train, dtype=np.float32) 
t_train = torch.tensor(np.tile(t_train_1[:, np.newaxis, np.newaxis], (1, dp_train, dv_train)), dtype=torch.float)

dt_val, dp_val, dv_val = x_validate.size()
t_validate_1 = np.linspace(0, dt_val-1 , dt_val, dtype=np.float32)
t_validate = torch.tensor(np.tile(t_validate_1[:, np.newaxis, np.newaxis], (1, dp_val, dv_val)), dtype=torch.float)

dt_test, dp_test, dv_test = x_test.size()
t_test_1 = np.linspace(0, dt_test-1, dt_test, dtype=np.float32)  
t_test = torch.tensor(np.tile(t_test_1[:, np.newaxis, np.newaxis], (1, dp_test, dv_test)), dtype=torch.float)


value = 0
y_train,y_validate, y_test = x_train[...,value], x_validate[...,value], x_test[...,value]


###

save_path = '/Users/jaschob/Desktop/DoseAI/data/'

with open(save_path+"data_train_cancer_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump((t_train, x_train,y_train), datei)
                                
with open(save_path+"data_val_cancer_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump((t_validate, x_validate,y_validate), datei)
                        
with open(save_path+"data_test_cancer_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump((t_test, x_test,y_test), datei)     
    
    
    
with open(save_path+"data_val_cancer_value_u_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump(u_validate,datei)
        
with open(save_path+"data_test_cancer_value_u_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump(u_test,datei)
    
with open(save_path+"data_train_cancer_value_u_gamma_"+str(gamma)+".pkl", "wb") as datei:
    pickle.dump(u_train,datei)    
    
#####

training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)

def prep_map(data_map, device, unscaled= False):
    if unscaled:
        print(1)
        data_tensor=torch.from_numpy(data_map["unscaled_outputs"]).float().to(device)#um einen zp verschoben
        toxic_tensor = torch.from_numpy(data_map["unscaled_outputs_toxicity"]).float().to(device)
    else:
        print(2)
        data_tensor=torch.from_numpy(data_map["outputs"]).float().to(device)#um einen zp verschoben
        toxic_tensor = torch.from_numpy(data_map["outputs_toxicity"]).float().to(device)
        
    treatments_tensor = torch.from_numpy(data_map["current_treatments"]).float().to(device)
    covariables_x = torch.from_numpy(np.concatenate([data_map["current_covariates"][:,:-1,:],data_map["previous_treatments"]],axis=2)).float().to(device)
    time_tensor = torch.from_numpy(data_map["time_covariates"]).float().to(device)
    covariables_tensor = torch.cat([covariables_x,time_tensor[:,:-1,:]],dim=2)
    active_tensor=torch.from_numpy(data_map["active_entries"]).float().to(device)
    return data_tensor, toxic_tensor, treatments_tensor, covariables_tensor, time_tensor, active_tensor

device = 'cpu'
unscaled = True
if training_processed is not None:
    train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = prep_map(training_processed, device,unscaled)
if validation_processed is not None:
    validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = prep_map(validation_processed,device,unscaled)
if test_processed is not None:  
    test_X, test_toxic, test_treatments, covariables_x_test, test_time_covariates, active_entries_test = prep_map(test_processed, device,unscaled)
    


x_train_n = covariables_x.transpose(0,1)[...,[0,2,1,5]]
x_validate_n = covariables_x_val.transpose(0,1)[...,[0,2,1,5]]
x_test_n = covariables_x_test.transpose(0,1)[...,[0,2,1,5]]

y_train_n = covariables_x.transpose(0,1)[...,[0]]
y_validate_n = covariables_x_val.transpose(0,1)[...,[0]]
y_test_n = covariables_x_test.transpose(0,1)[...,[0]]

u_train_n = covariables_x.transpose(0,1)[...,[3,4]]
u_validate_n = covariables_x_val.transpose(0,1)[...,[3,4]]
u_test_n = covariables_x_test.transpose(0,1)[...,[3,4]]

t_train_n = torch.cat([t_train[:x_train_n.size(0)],t_train[:x_train_n.size(0)]],dim=2)
t_validate_n = torch.cat([t_validate[:x_validate_n.size(0)],t_validate[:x_validate_n.size(0)]],dim=2)
t_test_n = torch.cat([t_test[:x_test_n.size(0)],t_test[:x_test_n.size(0)]],dim=2)

x_u_mean = torch.tensor([training_processed['input_means'][0],
                       training_processed['input_means'][2],
                       training_processed['input_means'][1],
                       #training_processed['input_means'][5],
                       training_processed['input_means'][3],
                       training_processed['input_means'][4]])

x_u_sdt = torch.tensor([training_processed['inputs_stds'][0],
                       training_processed['inputs_stds'][2],
                       training_processed['inputs_stds'][1],
                       #training_processed['inputs_stds'][5],
                       training_processed['inputs_stds'][3],
                       training_processed['inputs_stds'][4]])





with open(save_path+"data_train_cancer_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump((t_train_n, x_train_n,y_train_n), datei)
                                
with open(save_path+"data_val_cancer_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump((t_validate_n, x_validate_n,y_validate_n), datei)
                        
with open(save_path+"data_test_cancer_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump((t_test_n, x_test_n,y_test_n), datei)     
    
    
    
with open(save_path+"data_val_cancer_value_u_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump(u_validate_n,datei)
        
with open(save_path+"data_test_cancer_value_u_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump(u_test_n,datei)
    
with open(save_path+"data_train_cancer_value_u_gamma_"+str(gamma)+"_n.pkl", "wb") as datei:
    pickle.dump(u_train_n,datei)    
    
