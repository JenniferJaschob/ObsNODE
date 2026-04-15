#https://github.com/philippwendland/OptAB/blob/main/Code/OptAB/training_encoder.py

#Info #binäre Behandlungen (bei der Vorbverarbeitung mit Bool-Val umstellen)
import pickle
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_path = '/Users/jaschob/Desktop/data_m4/data_preproc_phil/data_prep_three/' #phil Vorverarbeitung

    
with open(save_path+'sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)
# Tensor of patient data, size: number of patients x timepoints x variables (including one dimension corresponding to time)
 
with open(save_path+'sepsis_all_1_keys.pkl', 'rb') as handle:
    key_times = pickle.load(handle)
# Dictionary where keys correspond to the index and the values correspond to the time in hours

with open(save_path+'sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)
# List of all variable names of the time-dependent variables

with open(save_path+'sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)
# Tensor of static variables, size: number of patients x static_variables

with open(save_path+'sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)
# List of all static variable names

with open(save_path+'sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)
# list of means of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)

with open(save_path+'sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
# list of standard deviations of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)
    
with open(save_path+'sepsis_all_1_indices_train.pkl', 'rb') as handle:
    indices_train = pickle.load(handle)
    
with open(save_path+'sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)
# list of index variables for the variables to be standardized

with open(save_path+'sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)

# Threshold for the model to compute only positive outputs (via softplus)
data_thresh = ((0-variables_mean)/variables_std)[[variables.index("SOFA"),variables.index("creatinine"),variables.index("bilirubin_total"),variables.index("alt")]]

rectilinear_index=0

#### "Pre-Processing of data"
# "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_train])
key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times={list(key_times.keys())[x]: key_times[x] for x in key_times_index}

#Extracting outcome and side-effects from data and setting device
data_X = data[:,list(key_times.values())[1:],1:2][indices_train].to(device)
data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic=data_toxic[:,:,None][indices_train].to(device)
data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2=data_toxic2[:,:,None][indices_train].to(device)
data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3=data_toxic3[:,:,None][indices_train].to(device)
data_toxic = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)

#Extracting treatments and side-effects from data and setting device
data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment=data_treatment[:,:,None][indices_train].to(device)
data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2=data_treatment2[:,:,None][indices_train].to(device)
data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3=data_treatment3[:,:,None][indices_train].to(device)
data_treatment = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)

#Extracting the covariables
data_covariables = data[:,:list(key_times.keys())[-1],:].clone()[indices_train].to(device)

#Normalizing the missing masks to  one
time_max = data.shape[1]
data_covariables[:,:,len(variables)+1:] = data_covariables[:,:,len(variables)+1:]/time_max
data_covariables[:,:,0] = data_covariables[:,:,0]/time_max

# Selection of training and test data
data_time = data[:,:list(key_times.keys())[-1],0:1][indices_train].to(device)
data_active = ~data[:,list(key_times.values()),1:2].isnan()[indices_train].to(device)
data_static=static_tensor[indices_train].to(device,dtype=torch.float32)

train_index = np.random.choice(a=list(range(data_X.shape[0])),size=int(data_X.shape[0]*0.8),replace=False)
val_index = [i for i in list(range(data_X.shape[0])) if i not in indices_train]


# Train/Val Split for training
data_X_train =data_X[train_index]#1
data_toxic_train=data_toxic[train_index]#3
data_treatment_train=data_treatment[train_index]#3
data_covariables_train = data_covariables[train_index]#26
data_time_train=data_time[train_index]
data_active_train = data_active[train_index]
data_static_train = data_static[train_index]

data_static_train_expanded = data_static_train.unsqueeze(1).expand(-1, data_X_train.size(1), -1) 
data_train = torch.cat([data_X_train,data_toxic_train,data_covariables_train,data_static_train_expanded ],axis=-1)
u_train = data_treatment_train


data_X_val =data_X[val_index]#Endpunkt SoferScore
data_toxic_val=data_toxic[val_index]#anderen Endpunkte
data_treatment_val=data_treatment[val_index]#Behandlungen u
data_covariables_val = data_covariables[val_index]#Kovariablen
data_time_val=data_time[val_index]
data_active_val = data_active[val_index]#MissingMaske
data_static_val = data_static[val_index]#Statische Variablen

data_static_val_expanded = data_static_val.unsqueeze(1).expand(-1, data_X_val.size(1), -1) 
data_val = torch.cat([data_X_val,data_toxic_val,data_covariables_val,data_static_val_expanded ],axis=-1)
u_val = data_treatment_val

###############################################################################

# ## "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)

# #Extracting outcome and side-effects from data and setting device
data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test].to(device)

data_toxic_t=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic_t=data_toxic_t[:,:,None][indices_test].to(device)
data_toxic2_t=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2_t=data_toxic2_t[:,:,None][indices_test].to(device)
data_toxic3_t=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3_t=data_toxic3_t[:,:,None][indices_test].to(device)
data_toxic_test = torch.cat([data_toxic_t,data_toxic2_t,data_toxic3_t],axis=-1)

###new
#Extracting treatments and side-effects from data and setting device
data_treatment_t=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment_t=data_treatment_t[:,:,None][indices_test].to(device)
data_treatment2_t=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2_t=data_treatment2_t[:,:,None][indices_test].to(device)
data_treatment3_t=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3_t=data_treatment3_t[:,:,None][indices_test].to(device)

###
# #Extracting treatments and side-effects from data and setting device
data_treatment_test = torch.cat([data_treatment_t,data_treatment2_t,data_treatment3_t],axis=-1)



#Extracting the covariables
data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test].to(device)

#Normalizing the missing masks to one
time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max

# Selection of training and test data
data_time_test = data[:,:list(key_times.keys())[-1],0:1][indices_test].to(device)
data_active_test = ~data[:,list(key_times.values()),1:2].isnan()[indices_test].to(device)
data_static_test=static_tensor[indices_test].to(device,dtype=torch.float32)    

# Compute unscaled data
data_toxic_test_unscaled=data_toxic_test.clone()
data_toxic_test_unscaled[:,:,0] = data_toxic_test_unscaled[:,:,0]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
data_toxic_test_unscaled[:,:,1] = data_toxic_test_unscaled[:,:,1]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
data_toxic_test_unscaled[:,:,2] = data_toxic_test_unscaled[:,:,2]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]


data_X_test_unscaled=data_X_test.clone()
data_X_test_unscaled[:,:,0] = data_X_test_unscaled[:,:,0]*variables_std[variables.index('SOFA')]+variables_mean[variables.index('SOFA')]

unscaled=False

data_X_test =data_X_test
data_toxic_test=data_toxic_test
#data_treatment_test=data_treatment_test
data_covariables_test = data_covariables_test
#data_time_test = data_time_test
data_active_test = data_active_test
data_static_test = data_static_test

data_static_test_expanded = data_static_test.unsqueeze(1).expand(-1, data_X_test.size(1), -1) 
data_test = torch.cat([data_X_test,data_toxic_test,data_covariables_test,data_static_test_expanded ],axis=-1)

data_test_unscaled_0to4 = torch.cat([data_X_test_unscaled,data_toxic_test_unscaled,data_covariables_test,data_static_test_expanded ],axis=-1)
#u_test = data_treatment_test

data_time_val_expanded = data_time_val.expand(-1, -1, data_val.size(-1))
data_time_train_expanded = data_time_train.expand(-1, -1, data_train.size(-1))
data_time_test_expanded = data_time_test.expand(-1, -1, data_test.size(-1))

# path_data = '/Users/jaschob/Desktop/save_mimiciv_pkl/'
# with open(path_data +"data_val_mimiciv2.pkl", "wb") as datei:
#     pickle.dump((data_time_val_expanded.transpose(0, 1),data_val.transpose(0, 1),data_X_val.transpose(0, 1)), datei)#t_validate, x_validate,y_validate
    
# with open(path_data+"data_train_mimiciv2.pkl", "wb") as datei:
#     pickle.dump((data_time_train_expanded.transpose(0, 1),data_train.transpose(0, 1),data_X_train.transpose(0, 1)), datei)

# with open(path_data+"data_test_mimiciv2.pkl", "wb") as datei:
#     pickle.dump((data_time_test_expanded.transpose(0, 1),data_test.transpose(0, 1),data_X_test.transpose(0, 1)), datei)

# # Steuerung u    
# with open(path_data + "data_val_mimiciv_value_u2.pkl", "wb") as datei:
#     pickle.dump((data_treatment_val.transpose(0, 1)), datei)
    
# with open(path_data+"data_train_mimiciv_value_u2.pkl", "wb") as datei:
#     pickle.dump((data_treatment_train.transpose(0, 1)), datei)

# with open(path_data+"data_test_mimiciv_value_u2.pkl", "wb") as datei:
#     pickle.dump((data_treatment_test.transpose(0, 1)), datei)
    
    
###############################################################################
# train_output=data_X_train
# train_toxic=data_toxic_train
# train_treatments=data_treatment_train
# covariables=data_covariables_train
# active_entries=data_active_train
# validation_output=data_X_test
# validation_toxic=data_toxic_test
# validation_treatments=data_treatment_test
# covariables_val=data_covariables_test
# active_entries_val=data_active_test
# static=data_static_train
# static_val=data_static_test
# rectilinear_index=rectilinear_index
# train_coeffs =  covariables

# # Initialize data loader
# if static is not None:
#     train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments, static)
# else:
#     train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_output, active_entries, train_toxic, train_treatments)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

# ###############################################################################
    
    
# path_data = '/Users/jaschob/Desktop/save_mimiciv_csv/'
# with open(path_data + "data_val_mimiciv.pkl", "wb") as datei:
#     pickle.dump((data_time_val_expanded, data_val, data_X_val),
#                 datei)  # t_validate, x_validate,y_validate

# with open(path_data+"data_train_mimiciv.pkl", "wb") as datei:
#     pickle.dump((data_time_train_expanded, data_train, data_X_train), datei)

# with open(path_data+"data_test_mimiciv.pkl", "wb") as datei:
#     pickle.dump((data_time_test_expanded, data_test, data_X_test), datei)

# # Steuerung u
# with open(path_data + "data_val_mimiciv_value_u.pkl", "wb") as datei:
#     pickle.dump((data_treatment_val), datei)

# with open(path_data+"data_train_mimiciv_value_u.pkl", "wb") as datei:
#     pickle.dump((data_treatment_train), datei)

# with open(path_data+"data_test_mimiciv_value_u.pkl", "wb") as datei:
#     pickle.dump((data_treatment_test), datei)
    

