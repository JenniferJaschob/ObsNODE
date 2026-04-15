import torch
import numpy as np
from src.utils.data_utils import process_data, read_from_file
import utils_paper_cancer as utils

print(torch.cuda.is_available())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_seed_i = 1
print(num_seed_i)
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)
    
save = str('Trained_Encoder_seed_'+str(num_seed_i)+'doseai.pth')
save2 = str('Trained_Decoder_seed_'+str(num_seed_i)+'doseai.pth')

transformed_datapath =  '/Users/jaschob/Desktop/DoseAI/data_dict.p'
pickle_map = read_from_file(transformed_datapath)
training_processed, validation_processed, test_processed = process_data(pickle_map,toxicity=True,continuous=True)
    
treatment_options = 2
    
    
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
    
model = utils.NeuralCDE(input_channels=6, hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=2, treatment_options=2, activation = activation, num_depth=num_depth, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth)
model.load_state_dict(torch.load(save))
model=model.to(model.device)
    
if training_processed is not None:
    train_X, train_toxic, train_treatments, covariables_x, time_covariates, active_entries = utils.prep_map(training_processed, model.device)
if validation_processed is not None:
    validation_X, validation_toxic, validation_treatments, covariables_x_val, validation_time_covariates, active_entries_val = utils.prep_map(validation_processed,model.device)
 
offset=0
max_horizon = 5
    
input_channels_dec=3
output_channels=2


hidden_channels_dec = 22
batch_size_dec = 125
hidden_states_dec = 802
lr_dec = 0.0016227982436909543
activation_dec = 'leakyrelu'
num_depth_dec = 13
pred_act_dec = 'leakyrelu'
pred_states_dec = 798
pred_depth_dec = 1
    
model_decoder = utils.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=hidden_channels,activation=activation_dec,num_depth=num_depth_dec, interpolation="linear",continuous=True, treat_thresh=treat_thresh, pos=True, thresh=thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec)
model_decoder=model_decoder.to(model_decoder.device)
    
loss = utils.train_dec_offset(model = model,
                       model_decoder = model_decoder,
                       train_output = train_X,
                       train_toxic = train_toxic,
                       train_treatments = train_treatments,
                       covariables = covariables_x,
                       time_covariates = time_covariates,
                       active_entries = active_entries,
                       validation_output = validation_X,
                       validation_toxic= validation_toxic,
                       validation_treatments= validation_treatments,
                       covariables_val= covariables_x_val,
                       validation_time_covariates = validation_time_covariates,
                       active_entries_val = active_entries_val,
                       offset=offset,
                       max_horizon=max_horizon,
                       lr=lr_dec,
                       batch_size=batch_size_dec,
                       patience=10,
                       delta=0.0001,
                       max_epochs=1000,
                       #hypopt=True,
                       a_loss="spearman",
                       early_stop_path=save2,
                       static = None,
                       static_val = None)



torch.save(model_decoder.state_dict(), save2)



