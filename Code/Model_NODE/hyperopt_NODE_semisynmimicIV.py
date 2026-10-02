import torch
import numpy as np
import optuna
import joblib
import pickle

#from utils_observableNODE import *
import os
os.chdir('/home/jaschob/server/NODE/')
from utils_NODE import *

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1)
np.random.seed(1)

def time_objective(trial):

    #### read data ####
    normalize = True
    path_data = '/home/jaschob/server/semi_syn_mimic4/'
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

    num_time = 48
        
    t_validate, x_validate =    t_validate[:num_time].to(torch.float).to(device), x_validate[:num_time].to(torch.float).to(device)
    t_train, x_train =          t_train[:num_time].to(torch.float).to(device), x_train[:num_time].to(torch.float).to(device)
    t_test, x_test =            t_test[:num_time].to(torch.float).to(device), x_test[:num_time].to(torch.float).to(device)

    u_train, u_test, u_validate =  u_train[:num_time].to(torch.float).to(device),  u_test[:num_time].to(torch.float).to(device), u_validate[:num_time].to(torch.float).to(device)

    #if normalize:
    #    y_data_train, y_data_val, y_data_test = normalize_data(y_data_train, y_data_train), normalize_data(y_data_val, y_data_train), normalize_data(y_data_test, y_data_train)
    #    a_true_train, a_true_val, a_true_test = normalize_data(a_true_train, a_true_train), normalize_data(a_true_val, a_true_train), normalize_data(a_true_test, a_true_train)


    n_x = x_test.size(2)
    n_u = u_train.size(-1)
    n_t = 0
    
    n_x_dach = n_x
    w=0
    
    optimizer_func = 'Adam' 
    method = 'rk4'
    loss_op='default'
    patience = 30#100
    epochs = 50#150
    
    list_index_t_s = [12,24,36]
    step_size = 1.
    time_obs_pred=False
    save = '/home/jaschob/server/NODE/hyperopt_NODE_semisynmimicIV/out_NODE_semisynmimicIV'

    # Hyperparameter training 
    n_z = 4 #4 OLSN, 2 ObsNODE #trial.suggest_int('n_z',n_x,n_x+8)
    #batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    batch_size = trial.suggest_categorical('batch_size',[100,250,500,750,1000])
    learning_rate = trial.suggest_categorical('learning_rate',[1e-3,1e-4,1e-5])
    # Hyperparameter Observer
    hidden_dim_obs = trial.suggest_categorical('hidden_dim_obs',[32,64,128,256])
    
    # Hyperparameter NODE and helper NN (same)
    hidden_sizes_node = trial.suggest_categorical('hidden_dim_node',[32,64,128,256])

    num_layers_node = trial.suggest_int('num_layers_node',1,10)
    activation_node = trial.suggest_categorical('activation_node',['leakyrelu','tanh','sigmoid']) 
    
    # define Model ################################################################
    
    myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
        
    myNODE = MyNeuralODE_withNNs( activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
        
    model = MyModel_adjoint(myNODE, myObserver).to(device)

    try:
        
        out,loss = train_diff_ts(model_node=myNODE,
                                model_observer=myObserver,
                                u_train=u_train,u_val=u_validate,
                                time_obs_pred=time_obs_pred,
                                x_train=x_train,x_val=x_validate,
                                t_train=t_train,t_val=t_validate,
                                method=method,
                                patience=patience,
                                optimizer_func=optimizer_func,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size = batch_size,
                                list_index_t_s=list_index_t_s,
                                save_path=save,
                                step_size=step_size,
                                loss_op=loss_op,
                                hyperopt=True,
                                n_x_dach=n_x_dach,
                                w=w)  

        
    except Exception as e:
        print(e)
        loss = np.nan
        out = np.nan
        print(trial.number)
        print(loss)
        print(trial.params)
    
    torch.save(out, '/home/jaschob/server/NODE/hyperopt_NODE_semisynmimicIV/out_hyperopt_semisynmimicIV_for_trail_' + str(trial.number) + "NODE.pkl")

    import math

    if isinstance(loss, (float, np.floating)):
        return loss

    return loss[-1]
    
load_path = None
for i in range(30):    
    if load_path != None:
        study = joblib.load(load_path)
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = '/home/jaschob/server/NODE/hyperopt_NODE_semisynmimicIV/study_hyperopt_semisynmimicIV_for_trail_' +str(i)+'NODE.pkl'
    joblib.dump(study, load_path)

    
print("Best value:", study.best_value)
print("Best params:", study.best_params)