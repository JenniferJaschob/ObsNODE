
#generate semi Sinthetic MIMIC4 Data
from semi_synthetic_dataset_MimicIV import MIMIC4SyntheticDatasetCollection,MIMIC4SyntheticDataset

import pickle



path = ''
synth_treatments_list=['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']
treatment_outcomes_influence= {'Sofa': 1.,
                               'creatinine':1.,
                               'bilirubin_total':1.,
                               'alt':1.,
                               'Vancomycin':1.,
                               'Piperacillin-Tazobactam':1.,
                               'Ceftriaxon':1.}
min_seq_length= 20
max_seq_length= 100
max_number= 200
seed= 100
data_seed = 100
split = {'train':0.8,'val': 0.2, 'test': 0.2}
projection_horizon=10
autoregressiv = True
n_treatments_seq=10
treatment_sequence= ['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']


path_data = '/Users/jaschob/Desktop/save_mimiciv_csv/'
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

t_validate, x_validate, y_validate  =  data_validate
t_train, x_train, y_train           =  data_train
t_test, x_test, y_test              =  data_test

val_list= [5,9,14,15]+[4,6,7,8,10,11,12,13]+list(range(19,29))+list(range(30,37))+[38,39,40]

val_list_dynamic= [5,9,14,15]+list(range(4,13))+list(range(19,29))
val_list_static= list(range(30,37))+[38,39,40]
num_time = 12

x_validate_dynamic,x_validate_static =             x_validate[...,val_list_dynamic], x_validate[...,val_list_static]
x_train_dynamic, x_train_static =           x_train[...,val_list_dynamic], x_train[...,val_list_static]
x_test_dynamic,x_test_static =             x_test[...,val_list_dynamic], x_test[...,val_list_static]

"""
        Args:
            path: Path with MIMIC-4 dataset (HDFStore)
            synth_outcomes_list: List of SyntheticOutcomeGenerator
            synth_treatments_list: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            max_number: Maximum number of patients in cohort
            seed: Seed for sampling random functions
            data_seed: Seed for random cohort patient selection
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
            treatment_sequence: nested list of treatments to train on
"""
MIMIC4SyntheticDatasetCollection(path=path,
                 synth_outcomes_list=synth_outcomes_list,
                 synth_treatments_list=synth_treatments_list,
                 treatment_outcomes_influence=treatment_outcomes_influence,
                 min_seq_length=min_seq_length,
                 max_seq_length=max_seq_length,
                 max_number=max_number,
                 seed=seed,
                 data_seed=data_seed,
                 split=split,
                 projection_horizon=projection_horizon,
                 autoregressive=autoregressiv,
                 n_treatments_seq=n_treatments_seq,
                 treatment_sequence=treatment_sequence,
                 x_train = x_train,
                 x_val = x_validate,
                 x_test = x_test)
                 
#ruft MIMIC4SyntheticDataset auf
"""
        Args:
            all_vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            synthetic_outcomes: List of SyntheticOutcomeGenerator
            synthetic_treatments: List of SyntheticTreatment
            treatment_outcomes_influence: dict with treatment-outcomes influences
            subset_name: train / val / test
            mode: factual / counterfactual_one_step / counterfactual_treatment_seq
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            treatments_seq: Fixed (non-random) treatment sequecne for multiple-step-ahead prediction
            n_treatments_seq: Number of random trajectories after rolling origin in test subset
"""
train_f = MIMIC4SyntheticDataset(all_vitals=x_train_dynamic,
                                 static_features=x_train_static,
                                 synthetic_outcomes=synth_outcomes_list,
                                 synthetic_treatments=synth_treatments_list,
                                 treatment_outcomes_influence=treatment_outcomes_influence,
                                 subset_name='train')

val_f = MIMIC4SyntheticDataset(all_vitals=x_validate_dynamic,
                               static_features=x_validate_static,
                               synthetic_outcomes=synth_outcomes_list,
                               synthetic_treatments=synth_treatments_list,
                               treatment_outcomes_influence=treatment_outcomes_influence,
                               subset_name='val')

test_f = MIMIC4SyntheticDataset(all_vitals=x_test_dynamic,
                                static_features=x_test_static,
                                synthetic_outcomes=synth_outcomes_list,
                                synthetic_treatments=synth_treatments_list,
                                treatment_outcomes_influence=treatment_outcomes_influence,
                                subset_name='test')

