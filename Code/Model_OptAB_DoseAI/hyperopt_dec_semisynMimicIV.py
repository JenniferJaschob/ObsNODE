import torch
import numpy as np
import utils_paper_MimicIV as utils_paper

import optuna
import joblib
import pandas as pd
import pickle


print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_seed_i = 1
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

def time_objective(trial):
    #Data
    path_data = '/home/jaschob/server/semi_syn_mimic4/'
    save = '/home/jaschob/server/OptAB/res_semi_syn_mimic4/'
    
    with open(path_data + "x_validate_syn.pkl", "rb") as datei:
        x_validate = pickle.load(datei)        
    with open(path_data + "x_train_syn.pkl", "rb") as datei:
        x_train = pickle.load(datei)    
    with open(path_data + "x_test_syn.pkl", "rb") as datei:
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

    data_X_train = x_train[1:].transpose(0,1).to(device)
    data_covariables_train = torch.cat([x_train,u_train,t_train[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)
    data_treatment_train = u_train.transpose(0,1).to(device)
    data_active_train =   ~x_train.transpose(0,1).isnan().to(device)
    data_toxic_train = None

    data_X_test = x_validate[1:].transpose(0,1).to(device)
    data_covariables_test = torch.cat([x_validate,u_validate,t_validate[...,0:1]],dim=-1)[:-1].transpose(0,1).to(device)
    data_treatment_test = u_validate.transpose(0,1).to(device)
    data_active_test =  ~x_validate.transpose(0,1).isnan().to(device)
    data_toxic_test = None
    
    
    data_time_train = t_train[:-1,:,0:1].transpose(0,1).to(device)
    data_time_test = t_validate[:-1,:,0:1].transpose(0,1).to(device)
    

    hidden_channels = 7
    batch_size = 1000
    hidden_states = 174
    lr = 0.0015972840572993194
    activation = 'leakyrelu'
    num_depth = 14
    pred_act = 'leakyrelu'
    pred_states = 275
    pred_depth = 4
    
    pred_comp=True
    
    # Hyperparameters and their distributions
    hidden_channels_dec = trial.suggest_int('hidden_channels_dec',1,30)
    batch_size_dec=trial.suggest_categorical('batch_size_dec',[1000,2000])
    hidden_states_dec = trial.suggest_int('hidden_states_dec',16,1000)
    lr_dec = trial.suggest_uniform('lr_dec',0.0001,0.01)
    activation_dec = trial.suggest_categorical('activation_dec',['leakyrelu','tanh','sigmoid','identity'])
    num_depth_dec = trial.suggest_int('numdepth_dec',1,20)
    
    pred_comp=True
    pred_act_dec = trial.suggest_categorical('pred_act_dec',['leakyrelu','tanh','sigmoid','identity'])
    pred_states_dec = trial.suggest_int('pred_states_dec',16,1000)
    pred_depth_dec = trial.suggest_int('preddepth',1,6)
    
    
    # Initializing and loading the Encoder
    data_thresh = torch.zeros(1)
    model = utils_paper.NeuralCDE(input_channels=data_covariables_train.size(-1),
                                  hidden_channels=hidden_channels,
                                  hidden_states=hidden_states,
                                  output_channels=1,#1,
                                  treatment_options=data_treatment_train.size(-1),
                                  activation = activation,
                                  num_depth=num_depth,
                                  interpolation="linear",
                                  pos=True,
                                  thresh=data_thresh,
                                  pred_comp=pred_comp,
                                  pred_act=pred_act,
                                  pred_states=pred_states,
                                  pred_depth=pred_depth,
                                  device=device)
    
    model=model.to(model.device)
    model.load_state_dict(torch.load(save+'final_model_semisynmimic4_19.pth'))
    
    offset=0

    # Initializing Decoder
    rectilinear_index=0
    z0_hidden_dimension_dec = hidden_channels+5

    model_decoder = utils_paper.NeuralCDE(input_channels=data_X_train.size(-1),#1
                                          hidden_channels=hidden_channels_dec,
                                          hidden_states=hidden_states_dec,
                                          output_channels=1,
                                          z0_dimension_dec=z0_hidden_dimension_dec,
                                          activation=activation_dec,
                                          num_depth=num_depth_dec,
                                          pos=True,
                                          thresh=data_thresh,
                                          pred_comp=True,
                                          pred_act=pred_act_dec,
                                          pred_states=pred_states_dec,
                                          pred_depth=pred_depth_dec,
                                          treatment_options=data_treatment_train.size(-1),
                                          device=device)#2

    model_decoder=model_decoder.to(device)
    
    max_horizon=48#121
    b=list(range(max_horizon))
    offset_train_list=[]
    for i in range(0,len(b),10):
        if i<=20:
            c=np.random.choice(b[i:i+10],size=2,replace=False)
            offset_train_list.append(c[0])
            offset_train_list.append(c[1])
        else:
            offset_train_list.append(np.random.choice(b[i:i+10],replace=False))
    offset_train_list.sort()
    
    # Training for specific hyperparameterconfigurations
    try:
        loss = utils_paper.train_dec_offset(model,
                                            model_decoder,
                                            offset=offset,
                                            max_horizon=max_horizon,
                                            lr=lr_dec,
                                            batch_size=batch_size_dec,
                                            patience=10,
                                            delta=0.0001,
                                            max_epochs=1000,
                                            weight_loss=True,
                                            train_output=data_X_train,
                                            train_toxic=None,#data_toxic_train,
                                            train_treatments=data_treatment_train,
                                            covariables=data_covariables_train,
                                            time_covariates=data_time_train,
                                            active_entries=data_active_train,
                                            static=None,
                                            
                                            rectilinear_index=rectilinear_index,
                                            validation_output=data_X_test,
                                            validation_toxic=None,#data_toxic_test,
                                            validation_treatments=data_treatment_test,
                                            covariables_val=data_covariables_test,
                                            validation_time_covariates=data_time_test,
                                            active_entries_val=data_active_test,
                                            static_val=None,
                                            
                                            offset_train_list=offset_train_list,
                                            offset_val_list=None,
                                            early_stop_path=save+'out_early_dec.pth',
                                            dec_expand=True,
                                            sofa_expand=True,
                                            med_dec=False,
                                            med_dec_start=True)
    except Exception as e:
        print(e)
        loss = np.nan
    
    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model_decoder.state_dict(), save+'final_decoder_model_semisynmimic4_' + str(trial.number) + '.pth')
    return loss


load_path = None
for i in range(20):    
    if load_path != None:
        study = joblib.load(load_path)
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = '/home/jaschob/server/OptAB/res_semi_syn_mimic4/study_decoder_static_batch_for_trail_' +str(i)+'.pkl'
    joblib.dump(study, load_path)
    
    
print("Best value:", study.best_value)
print("Best params:", study.best_params)


