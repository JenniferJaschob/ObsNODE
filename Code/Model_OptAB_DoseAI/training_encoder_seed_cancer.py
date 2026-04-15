import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils 


import pandas as pd

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_seed_i = 1
print(num_seed_i)
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)
    
save = str('Trained_Encoder_seed_'+str(num_seed_i)+'doseai2.pth')


transformed_datapath =  '/Users/jaschob/Desktop/DoseAI/data_dict.p'
pickle_map = read_from_file(transformed_datapath)
training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)
    
#treatment_options = 2

hidden_channels = 16
batch_size = 250
hidden_states = 578
lr = 0.004239690693777566
activation = 'leakyrelu'
num_depth = 2
pred_act = 'tanh'
pred_states = 128
pred_depth = 4
pred_comp=True
    
thresh=torch.Tensor([(0-training_processed["output_means"])/training_processed["output_stds"],(0-training_processed["output_toxicity_means"])/training_processed["output_toxicity_stds"]])
treat_thresh=torch.Tensor([(0-training_processed["input_means"][2])/training_processed["inputs_stds"][2],(0-training_processed["input_means"][3])/training_processed["inputs_stds"][3]])    
    
#model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=treatment_options, activation = activation, num_depth=num_depth, interpolation="linear", continuous=True,treat_thresh=treat_thresh, pos=True, thresh=thresh)
model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)    
model=model.to(model.device)
    
if training_processed is not None:
    train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
if validation_processed is not None:
    validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)



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
                    a_loss='spearman',
                    early_stop_path=save)

torch.save(model.state_dict(), save)


torch.save(model, str('Trained_Encoder_seed_'+str(num_seed_i)+'doseai_model.pth'))  

