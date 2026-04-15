import torch
import numpy as np
import optuna
import joblib
import pickle

from utils_observableNODE import * 

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def time_objective(trial):

    
    path_data = 'server/data_mimic/'        
        
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

    num_time = 12
    
    t_validate, x_validate =    t_validate[:num_time,:,val_list].to(torch.float).to(device), x_validate[:num_time,:,val_list].to(torch.float).to(device)
    t_train, x_train =          t_train[:num_time,:,val_list].to(torch.float).to(device), x_train[:num_time,:,val_list].to(torch.float).to(device)
    t_test, x_test =            t_test[:num_time,:,val_list].to(torch.float).to(device), x_test[:num_time,:,val_list].to(torch.float).to(device)
    
    u_train[torch.isnan(u_train)],u_test[torch.isnan(u_test)],u_validate[torch.isnan(u_validate)] = 0,0,0
    u_validate, u_test, u_train = u_validate[:num_time,:,:].to(torch.float).to(device), u_test[:num_time,:,:].to(torch.float).to(device), u_train[:num_time,:,:].to(torch.float).to(device)
    
    ####
    
    n_x = x_test.size(2)
    n_u = u_train.size(-1)
    n_t = 0
    
    n_x_dach = 4
    w=0
    
    optimizer_func = 'Adam' 
    method = 'rk4'
    loss_op='default'
    patience = 30
    epochs = 50
    
    list_index_t_s = [4,6,8,10] 
    step_size = 1.
    time_obs_pred=False
    save = 'server/out_oss_hyperopt1'

    # # Hyperparameter training
    n_z = trial.suggest_int('n_z',n_x,n_x+3)
    batch_size=trial.suggest_categorical('batch_size',[100,250,500,750,1000])
    learning_rate = trial.suggest_categorical('learning_rate',[1e-3,1e-4,1e-5])
    # Hyperparameter Observer
    hidden_dim_obs = trial.suggest_categorical('hidden_dim_obs',[32,64,128,256])

    
    # Hyperparameter NODE and helper NN (same)
    hidden_sizes_node = trial.suggest_categorical('hidden_dim_node',[32,64,128,256])

    num_layers_node = trial.suggest_int('num_layers_node',5,10)
    activation_node = trial.suggest_categorical('activation_node',['leakyrelu','tanh','sigmoid']) 
    
    # define Model ################################################################
    myObserver= LSTMRecognitionModel_with_nan(n_z=n_z,n_x=n_x,n_t=n_t, hidden_dim=hidden_dim_obs).to(device)
    
    helperNN_list = [] 
    for j in range(1,n_z):
        helperNN= MyhelperNN(activation = activation_node ,hidden_sizes =hidden_sizes_node ,num_layers=num_layers_node,n_u=n_u,n_z_i=j*n_x,output_size=n_x).to(device)
        helperNN_list.append(helperNN)
        
    myNODE = MyObservableNeuralODE_withNNs(helperNN_list=helperNN_list, activation = activation_node, hidden_sizes=hidden_sizes_node,num_layers=num_layers_node,n_z=n_z,n_x=n_x,n_u=n_u).to(device)
    
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
        

        #loss =out['val_loss'][-1]
        
    except Exception as e:
        print(e)
        loss = np.nan
        
        print(trial.number)
        print(loss)
        print(trial.params)
    
    torch.save(out, '/home/jaschob/server/hyperopt_server_run8/out_hyperopt_mimic4_12_for_trail_' + str(trial.number) + "_run8.pkl")
    
    return loss[-1]
    
load_path = None

for i in range(30):    
    if load_path != None:
        study = joblib.load(load_path)#laed
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = '/home/jaschob/server/hyperopt_server_run8/study_hyperopt_mimic4_12_for_trail_' +str(i)+'_run8.pkl'
    joblib.dump(study, load_path)
    
    
print("Best value:", study.best_value)
print("Best params:", study.best_params)