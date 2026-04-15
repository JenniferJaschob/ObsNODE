#Adaptation from  github.com/philippwendland/DoseAI/blob/main/hypopt_encoder_cancer.py
# Maik Kschischo,Philipp Wendland. Counterfactual AI for Dynamic Dose Optimization with Side-Effect Constraints. 
# TechRxiv. June 12, 2025., DOI: 10.36227/techrxiv.174970492.29621159/v1

import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils

import optuna
import joblib
import pandas as pd


save_res = '/Users/jaschob/Desktop/DoseAI/data/'

path = '/Users/jaschob/Desktop/DoseAI/data/'

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_seed_i = 1
print(num_seed_i)
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

gamma = 4

def time_objective(trial):
    
    #reading data
    transformed_datapath = path+"data_dict_"+str(gamma)+"_"+str(gamma)+".p"
    pickle_map = read_from_file(transformed_datapath)
    training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)
    
    treatment_options = 2
    
    hidden_channels = trial.suggest_int('hidden_channels',1,30)
    batch_size=trial.suggest_categorical('batch_size',[16,32,64,125,250,500,1000])
    hidden_states = trial.suggest_int('hidden_states',16,1000)
    lr = trial.suggest_uniform('lr',0.0001,0.01)
    activation = trial.suggest_categorical('activation',['leakyrelu','tanh','sigmoid','identity'])
    num_depth = trial.suggest_int('numdepth',1,20)
    
    pred_act = 'tanh'
    pred_states = 128
    pred_depth = 4
    pred_comp = True
    
    thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
    treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])    
    
    model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=treatment_options, activation = activation, num_depth=num_depth, interpolation="linear", continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)    

    model=model.to(model.device)
    
    if training_processed is not None:
        train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
    if validation_processed is not None:
        validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
    
    done=False
    tries = 0
    while done == False:
        if tries > 3:
            done = True
            break
        try:
            loss = utils.train(model = model,
                               train_output=train_X,
                               train_toxic=train_toxic,
                               train_treatments=train_treatments,
                               covariables= covariables_x,
                               #time_covariates,
                               active_entries=active_entries,
                               validation_output=validation_X,
                               validation_toxic=validation_toxic,
                               validation_treatments=validation_treatments,
                               covariables_val=covariables_x_val,
                               #validation_time_covariates,
                               active_entries_val=active_entries_val,
                               lr=lr,
                               batch_size=batch_size,
                               patience=10,
                               delta=0.0001,
                               max_epochs=1000,
                               #hypopt=True,
                               a_loss='spearman')
            done=True
        except Exception as e:
            print(e)
            tries = tries + 1
            loss = np.nan
            
    print(trial.number)
    print(loss)
    print(trial.params)
    
    
    torch.save(model.state_dict(), save_res+'final_model_hypopt_canncer_gamma_'+str(gamma)+'_for_trail_' + str(trial.number) + "phil.pth")

    return loss
###
    
load_path = None
for i in range(20):    
    if load_path != None:
        study = joblib.load(load_path)
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = save_res+'study_hyperopt_cancer_gamma_'+str(gamma)+'_for_trail_' +str(i)+'phil.pkl'
    joblib.dump(study, load_path)
    

print("Best value:", study.best_value)
print("Best params:", study.best_params)
#####
