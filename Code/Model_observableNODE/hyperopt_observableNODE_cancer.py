import torch
import numpy as np
import optuna
import joblib
import pickle

from utils_observableNODE import *

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gamma = 2

torch.manual_seed(1)
np.random.seed(1)

def time_objective(trial):
    
    with open("server/data_save/data_cancer/data_val_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
        data_validate = pickle.load(datei)

    with open("server/data_save/data_cancer/data_test_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
        data_test = pickle.load(datei)

    with open("server/data_save/data_cancer/data_train_cancer_gamma_"+str(gamma)+".pkl", "rb") as datei:
        data_train = pickle.load(datei)

    with open("server/data_save/data_cancer/data_val_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_validate = pickle.load(datei)

    with open("server/data_save/data_cancer/data_test_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_test = pickle.load(datei)

    with open("server/data_save/data_cancer/data_train_cancer_value_u_gamma_"+str(gamma)+".pkl", "rb") as datei:
        u_train = pickle.load(datei)
        
    t_validate, x_validate, _  =  data_validate
    t_train, x_train, _        =  data_train
    t_test, x_test, _          =  data_test
    
    t_validate,t_train, t_test  = t_validate.to(torch.float).to(device),t_train.to(torch.float).to(device), t_test.to(torch.float).to(device).to(device)
    x_validate,x_train, x_test  = x_validate.to(torch.float).to(device),x_train.to(torch.float).to(device), x_test.to(torch.float).to(device).to(device)  

    u_validate,u_train, u_test  = u_validate.to(torch.float).to(device),u_train.to(torch.float).to(device), u_test.to(torch.float).to(device).to(device)
    
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
    
    list_index_t_s = [4,8,12,16,20] 
    step_size = 1.
    time_obs_pred=False
    save = 'server/out_caner_hyperopt_DoseAI'

    # # Hyperparameter training
    n_z = trial.suggest_int('n_z',n_x,n_x+3)
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
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

        
    except Exception as e:
        print(e)
        loss = np.nan
        
        print(trial.number)
        print(loss)
        print(trial.params)
    
    torch.save(out, '/home/jaschob/server/hyperopt_server_cancer_DoseAI/out_hyperopt_canncer_gamma_'+str(gamma)+'_for_trail_' + str(trial.number) + "b.pkl")
    
    return loss[-1]

    
load_path = None
for i in range(30):    
    if load_path != None:
        study = joblib.load(load_path)
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = '/home/jaschob/server/hyperopt_server_cancer_DoseAI/study_hyperopt_cancer_gamma_'+str(gamma)+'_for_trail_' +str(i)+'b.pkl'
    joblib.dump(study, load_path)

    
print("Best value:", study.best_value)
print("Best params:", study.best_params)