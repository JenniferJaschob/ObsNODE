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
    
    path_data = '/home/jaschob/server/semi_syn_mimic4/'#'/Users/jaschob/Desktop/semi_syn_mimic4/'
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
    
    
    # Hyperparameter and their distributions
    hidden_channels = trial.suggest_int('hidden_channels',1,30)
    batch_size=trial.suggest_categorical('batch_size',[500,1000,2000])
    hidden_states = trial.suggest_int('hidden_states',16,1000)
    lr = trial.suggest_uniform('lr',0.0001,0.01)
    activation = trial.suggest_categorical('activation',['leakyrelu','tanh','sigmoid','identity'])
    num_depth = trial.suggest_int('numdepth',1,20)
    
    pred_comp=True
    pred_act = trial.suggest_categorical('pred_act',['leakyrelu','tanh','sigmoid','identity'])
    pred_states = trial.suggest_int('pred_states',16,1000)
    pred_depth = trial.suggest_int('preddepth',1,6)
    
    # Threshold for the model to compute only positive outputs (via softplus)
    data_thresh =  torch.zeros(1)
    
    # Initializing the Encoder
    #model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables))
    
   
    
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
                                  pred_depth=pred_depth)
    model=model.to(device)
    
    rectilinear_index=0
    
    ### End preprocessing
    
    # Training for specific hyperparameterconfigurations
    try:
        #loss = utils_paper.train(model, weight_loss=True, lr=lr, batch_size=batch_size, patience=10, delta=0.0001, max_epochs=1000, train_output=data_X_train, train_toxic=data_toxic_train, train_treatments=data_treatment_train, covariables=data_covariables_train, active_entries=data_active_train, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables_val=data_covariables_test, active_entries_val=data_active_test, static=data_static_train,static_val=data_static_test,rectilinear_index=rectilinear_index,early_stop_path=str('compdepth_static_batch' + str(trial.number) + '.pth'))
        loss = utils_paper.train(model,
                                 weight_loss=True,
                                 lr=lr,
                                 batch_size=batch_size,
                                 patience=10,
                                 delta=0.0001,
                                 max_epochs=1000,
                                 train_output=data_X_train,
                                 train_toxic=data_toxic_train,
                                 train_treatments=data_treatment_train,
                                 covariables=data_covariables_train,
                                 active_entries=data_active_train,
                                 
                                 validation_output=data_X_test,
                                 validation_toxic=data_toxic_test,
                                 validation_treatments=data_treatment_test,
                                 covariables_val=data_covariables_test,
                                 active_entries_val=data_active_test,
                                 #static=data_static_train,
                                 #static_val=data_static_test,
                                 rectilinear_index=0,
                                 early_stop_path=save+'out_early.pth')
        
    except Exception as e:
        print(e)
        loss = np.nan

    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model.state_dict(), save+'final_model_semisynmimic4_' + str(trial.number) + '.pth')
    
    return loss



####

load_path = None
for i in range(20):    
    if load_path != None:
        study = joblib.load(load_path)
    else:
        study = optuna.create_study()    
        
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    load_path  = '/home/jaschob/server/OptAB/res_semi_syn_mimic4/study_static_batch_for_trail_' +str(i)+'.pkl'
    #'/home/jaschob/server/hyperopt_server_cancer_DoseAI/study_hyperopt_cancer_gamma_'+str(gamma)+'_for_trail_' +str(i)+'b.pkl'
    joblib.dump(study, load_path)
    

#print every Trail
#for trial in study.trials:
#    print(f"Trial {trial.number}: {trial.params}, Wert: {trial.value}")
    
print("Best value:", study.best_value)
print("Best params:", study.best_params)

###    
    
# study = optuna.create_study()

# # n_trials is the stopping criterion, n_jobs is the number of parallel used cpus/gpus
# study.optimize(time_objective, n_trials=20, n_jobs=1)


# print("Number of finished trials: ", len(study.trials))
    
# print("Best trial:")
# trial = study.best_trial

# train_dir="/work/wendland/tecde_three_compdepth"

# print("  rec_loss: ", trial.value)

# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))

#     file = open(train_dir+'/tries_lat_static.txt', 'a')
#     file.write("    {}: {}".format(key, value))
#     file.write("\n")
# file.write(str(trial.user_attrs))
# file.write("rec_loss: "+str(trial.value))
# file.write("\n"+"------------"+"\n")
# file.close()

# dic = dict(trial.params)
# dic['value'] = trial.value

# df = pd.DataFrame.from_dict(data=dic,orient='index').to_csv(train_dir + '/tries_lat_static.csv',header=False)

# # Saving and exporting study
# joblib.dump(study, '/home/jaschob/server/OptAB/res_semi_syn_mimic4/study_static_batch.pkl')


