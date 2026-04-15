import logging
import hydra
import torch
import pandas as pd
import pickle
import numpy as np
import json
import sys
import copy
import os

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

os.chdir('/Users/jaschob/Desktop/G_Transformer_Übergabe/1_Python_Scripts_G_Transformer/')
from src.models.utils import FilteringMlFlowLogger
from Heatmaps.my_vergleichsmasse import my_compute_prediction, heatmap_pred

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for GT (G-Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False) # turn of strict mode
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True) # custom resolver: add interpolation expression for within config files
    
    if args.model.gt.tune_hparams and "coeff" not in args.dataset:
        args.dataset.coeff = 2.0

    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True)) # convert config object to yaml-formatted string. Resolve interpolation expressions. Log yaml

    # Initialisation of data
    seed_everything(args.exp.seed) # global seed

    dataset_collection = instantiate(args.dataset, _recursive_=True) # Instantiate dataset dynamically from dataset config

    # Inject other datasets into the dataset collection
    # train_data = read_csv_to_dict(to_absolute_path("runnables/training_data.csv"))
    # val_data = read_csv_to_dict(to_absolute_path("runnables/validation_data.csv"))
    # test_data = read_csv_to_dict(to_absolute_path("runnables/test_data.csv"))
    # dataset_collection.train_f.data = train_data
    # dataset_collection.val_f.data = val_data
    # dataset_collection.test_f.data = test_data
    # print("SCALING PARAMETERS", dataset_collection.train_scaling_params)
    # print("SCALING PARAMETERS", type(dataset_collection.train_scaling_params))


    # # === Save all unprocessed datasets === #
    # save_raw_dataset_to_csv(dataset_collection.train_f, "unprocessed_train")
    # save_raw_dataset_to_csv(dataset_collection.val_f, "unprocessed_val")
    # save_raw_dataset_to_csv(dataset_collection.test_f, "unprocessed_test_f")
    # save_raw_dataset_to_csv(dataset_collection.test_cf_treatment_seq, "unprocessed_test_cf_treatment_seq")



    dataset_collection.process_data_multi() # prepare data set

    if args.dataset.name == "tumor_generator" and args.dataset.manual:
        MEAN_CANCER = dataset_collection.train_scaling_params[0]["cancer_volume"]
        STD_CANCER = dataset_collection.train_scaling_params[1]["cancer_volume"]

        MEAN_TOXICITY = dataset_collection.train_scaling_params[0]["toxicity"]
        STD_TOXICITY = dataset_collection.train_scaling_params[1]["toxicity"]

        DUMMY_VALUE = np.array([-MEAN_CANCER/STD_CANCER, -MEAN_TOXICITY/STD_TOXICITY]) 

    # print("Train patients:", len(dataset_collection.train_f))
    # print("Val patients:", len(dataset_collection.val_f))
    # print("Test patients:", len(dataset_collection.test_f) if hasattr(dataset_collection, 'test_f') else "n/a")
    
    # print_dict_shapes(dataset_collection.test_f.data, "test_f")
    # print_dict_shapes(dataset_collection.test_cf_treatment_seq.data, "test_cf_n_step")

    # === Save all processed datasets === #
    # save_dataset_to_csv(dataset_collection.train_f, "processed_train")
    # save_dataset_to_csv(dataset_collection.val_f, "processed_val")
    # save_dataset_to_csv(dataset_collection.test_f, "processed_test_f")
    # save_dataset_to_csv(dataset_collection.test_cf_treatment_seq, "processed_test_cf_treatment_seq")


    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0

    if not args.dataset.name == "tumor_generator":
        args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    else:
        args.model.dim_static_features = 0

    # Train_callbacks
    gt_callbacks = []

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri, run_name='0') # exclude submodels from logging
        gt_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None
    if args.model.gt.tune_hparams:
            # ============================== Hyperparameter Optimization ===========================================
        args.model.gt.projection_horizon = 0
        gtmodel = instantiate(args.model.gt, args, dataset_collection, _recursive_=False) 


        gtmodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                            callbacks=gt_callbacks, terminate_on_nan=True,
                            gradient_clip_val=args.model.gt.max_grad_norm)
        gtmodel.finetune(resources_per_trial=args.model.gt.resources_per_trial)
    else:
        seeds = [100,101,102,103,104]
        for seed in seeds:
            args.exp.seed = seed
            seed_everything(args.exp.seed)
            # ============================== 1-step ahead prediction ===========================================
            # BEFORE: gpus=eval(str(args.exp.gpus)) AFTER: gpus=0
            args.model.gt.projection_horizon = 0
            gtmodel = instantiate(args.model.gt, args, dataset_collection, _recursive_=False)  # initialize g-transformer

            
            gtmodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                        callbacks=gt_callbacks, terminate_on_nan=True,
                                        gradient_clip_val=args.model.gt.max_grad_norm)
            if os.path.exists(to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_0_{args.exp.seed}.pth")):
                gtmodel.load_state_dict(torch.load(to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_0_{args.exp.seed}.pth")))
                gtmodel.eval()
                print("LOADED MODEL")
            else:


                #! for debugging
                # train_dataloader = DataLoader(
                #     dataset_collection.train_f,
                #     batch_size=8, 
                #     shuffle=True,
                #     num_workers=1,
                # )
                
                gtmodel_trainer.fit(gtmodel) #, train_dataloaders=train_dataloader)
                torch.save(gtmodel.state_dict(), to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_0_{args.exp.seed}.pth"))
                print("SAVED MODEL")


            for t in range(1, args.dataset.projection_horizon+1):
                seed_everything(args.exp.seed)  
                # ============================== Train ===========================================
                # BEFORE: gpus=eval(str(args.exp.gpus)) AFTER: gpus=0
                args.model.gt.projection_horizon = t
                gtmodel = instantiate(args.model.gt, args, dataset_collection, _recursive_=False)

                gtmodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                        callbacks=gt_callbacks, terminate_on_nan=True,
                                        gradient_clip_val=args.model.gt.max_grad_norm)
                if os.path.exists(to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_{t}_{args.exp.seed}.pth")):
                    gtmodel.load_state_dict(torch.load(to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_{t}_{args.exp.seed}.pth")))
                    gtmodel.eval()
                    print("LOADED MODEL")
                else:
                    gtmodel_trainer.fit(gtmodel)
                    torch.save(gtmodel.state_dict(), to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_{t}_{args.exp.seed}.pth"))
                    print("SAVED MODEL")


        #EVALUATION
        models = []
        prediction_arrays = []
        ground_truth_arrays = []
        for s in range(len(seeds)):
            models_for_s = []
            trainers_for_s = []
            for t in range(args.dataset.projection_horizon+1):

                args.model.gt.projection_horizon = t
                model = instantiate(args.model.gt, args, dataset_collection, _recursive_=False)
                model_trainer = Trainer(gpus=0, logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                                callbacks=gt_callbacks, terminate_on_nan=True,
                                                gradient_clip_val=args.model.gt.max_grad_norm)
                model.load_state_dict(torch.load(to_absolute_path(f"TrainedModels/gtmodel_weights_{args.dataset.name}_{t}_{seeds[s]}.pth")))
                model.trainer = model_trainer
                model.eval()
                models_for_s.append(model)
                trainers_for_s.append(model_trainer)

            prediction_for_s = []
            for t in range(1,len(models_for_s)+1):
                predictions = get_predictions(dataset_collection.test_f, models_for_s, t, DUMMY_VALUE)
                prediction_for_s.append(torch.tensor(predictions, dtype=torch.float32))

                ground_truth = dataset_collection.test_f.data["outputs"][:,:len(models_for_s),:]
                ground_truth_arrays.append(torch.tensor(ground_truth, dtype=torch.float32))

            prediction_arrays.append(prediction_for_s)
            models.append(models_for_s)

        # prediction_arrays = []
        # ground_truth_arrays = []
        # for s in range(len(seeds)):
        #     prediction_for_s = []
        #     for t in range(1,args.dataset.projection_horizon+1):
        #         predictions = get_predictions(dataset_collection.test_f, models[s], t, DUMMY_VALUE)
        #         prediction_for_s.append(torch.tensor(predictions, dtype=torch.float32))

        #         ground_truth = dataset_collection.test_f.data["outputs"][:,:len(models),:]
        #         ground_truth_arrays.append(torch.tensor(ground_truth, dtype=torch.float32))
        #     prediction_arrays.append(prediction_for_s)
        n_timesteps = ground_truth_arrays[0].shape[1]
        list_index_t_s_pred = list(range(n_timesteps))
        heatmap_pred(x_pred_list=prediction_arrays, x_true=ground_truth_arrays, list_index_t_s_pred=list_index_t_s_pred, vmin=0, vmax=0.9, max_horizon=len(models[0]), index=0, save_link=to_absolute_path(f"Heatmaps/Cancer_Toxicity_Heatmaps/heatmap_cancer_volume_gamma_{args.dataset.coeff}.png"))
        heatmap_pred(x_pred_list=prediction_arrays, x_true=ground_truth_arrays, list_index_t_s_pred=list_index_t_s_pred, vmin=0, vmax=0.6, max_horizon=len(models[0]), index=1, save_link=to_absolute_path(f"Heatmaps/Cancer_Toxicity_Heatmaps/heatmap_toxicity_gamma_{args.dataset.coeff}.png"))


            # ============================== Test ===========================================
            # if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse

                # test_rmse = gtmodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
            # elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
            #     test_rmse = gtmodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
            # test_rmses = {f'{t+1}-step': test_rmse}
            # logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

            # decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

            # mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
            # results.update(decoder_results)

        # mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results

# === Function to save a dataset to CSV ===
def save_dataset_to_csv(dataset_obj, filename_prefix):
    if dataset_obj is None:
        return
    print(dataset_obj.data.keys())
    data = dataset_obj.data  # Assume .data is a dictionary
    df = pd.DataFrame({
        'current_covariates': [list(row) for row in data.get('current_covariates', [])],
        'outputs': [list(row) for row in data.get('outputs', [])],
        'prev_treatments': [list(row) for row in data.get('prev_treatments', [])],
        'current_treatments': [list(row) for row in data.get('current_treatments', [])],
        #'static_features': [list(row) for row in data.get('static_features', [])],
        'sequence_lengths': data.get('sequence_lengths', []),
        'active_entries': [list(row) for row in data.get('active_entries', [])],
    })

    output_path = f"{filename_prefix}_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset {filename_prefix} saved to {output_path}")


def save_raw_dataset_to_csv(dataset_obj, filename_prefix):
    if dataset_obj is None:
        print(f"No dataset for {filename_prefix}, skipping...")
        return

    data = dataset_obj.data  # Should be a dict
    n_patients = len(data['sequence_lengths'])  # Number of patients

    all_rows = []

    for i in range(n_patients):
        row = {}

        for key in data.keys():
            entry = data[key][i]

            if isinstance(entry, (np.ndarray, list)):
                # Save arrays/lists as JSON strings
                row[key] = json.dumps(np.array(entry).tolist())
            else:
                # Scalars (e.g., sequence_lengths) save normally
                row[key] = entry

        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    output_path = f"{filename_prefix}_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset {filename_prefix} saved to {output_path}")

def read_csv_to_dict(csv_file):
    """
    Reads a CSV file and reconstructs the original dictionary.

    Parameters:
        csv_file (str): The path to the CSV file.

    Returns:
        dict: The reconstructed dictionary.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize the dictionary
    reconstructed_dict = {}

    # Iterate over the columns of the DataFrame
    for column in df.columns:
        # Check if the column contains JSON strings
        try:
            # Attempt to parse the first value as JSON
            first_value = df[column].iloc[0]
            if isinstance(first_value, str) and first_value.startswith("[") and first_value.endswith("]"):
                # Convert JSON strings back to arrays/lists
                parsed_column = df[column].apply(json.loads).to_numpy()

                # Check if the parsed column is a list of lists and convert to a 2D NumPy array
                if isinstance(parsed_column[0], list):
                    reconstructed_dict[column] = np.array(parsed_column.tolist())
                else:
                    reconstructed_dict[column] = parsed_column
            else:
                # Otherwise, treat the column as scalar values
                reconstructed_dict[column] = df[column].to_numpy()
        except Exception as e:
            print(f"Error processing column {column}: {e}")
            reconstructed_dict[column] = df[column].to_numpy()

    return reconstructed_dict

def print_dict_shapes(data_dict, dict_name="dict"):
    print(f"--- Shape overview for {dict_name} ---")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: np.ndarray, shape={value.shape}")
        elif isinstance(value, list):
            if len(value) > 0 and hasattr(value[0], '__len__'):
                print(f"{key}: list, len={len(value)}, entry shape={np.array(value[0]).shape}")
            else:
                print(f"{key}: list, len={len(value)}")
        else:
            print(f"{key}: {type(value)}, len={len(value) if hasattr(value, '__len__') else 'n/a'}")

def get_predictions(dataset_collection, gtmodels, t, dummy_value):

    # t = timesteps_given-1 
    temp_dataset_collection = copy.deepcopy(dataset_collection)
    temp_dataset_collection.data = cut_data_till_timestep(dataset_collection.data, t, dummy_value)
    temp_dataset_collection.data["predictions"] = np.zeros((temp_dataset_collection.data["outputs"].shape[0], len(gtmodels)-(t-1), temp_dataset_collection.data["outputs"].shape[-1]))

    for step in range(len(gtmodels)-(t-1)):

        preds = gtmodels[step].get_predictions(temp_dataset_collection)
        preds = torch.tensor(preds, dtype=torch.float32)

        temp_dataset_collection.data["predictions"][:, step, :] = preds[:, t, :] if preds.ndim == 3 else preds[:,:]  # Ensure correct shape
    return temp_dataset_collection.data["predictions"]  # Return the predictions without the first t timesteps (which are given as input)


def autoregressive_rollout(dataset_collection, gtmodel, t, max_horizon, dummy_value):

    # t = timesteps_given-1 
    temp_dataset_collection = copy.deepcopy(dataset_collection)
    temp_dataset_collection.data = cut_data_till_timestep(dataset_collection.data, t, dummy_value)
    temp_dataset_collection.data["predictions"] = np.zeros_like(dataset_collection.data["outputs"])

    for step in range(t, max_horizon):

        preds = gtmodel.get_predictions(temp_dataset_collection)
        preds = torch.tensor(preds, dtype=torch.float32)

        temp_dataset_collection.data = insert_next_prediction(preds, temp_dataset_collection.data, dataset_collection.data, step)

    return temp_dataset_collection.data["predictions"][:,t:,:]  # Return the predictions without the first t timesteps (which are given as input)

def insert_next_prediction(predictions, data, true_data, t):
    data["predictions"][:, t, :] = predictions[:, t, :]
    if t < data["current_covariates"].shape[1]-1:
        data["current_covariates"][:, t+1 , :] = predictions[:,t,:]
        data["prev_outputs"][:, t+1 , :] = predictions[:,t,:]
        #data["outputs"][:, t+1, :] = true_data[:,t+1,:]
        data["active_entries"][:, t+1] = np.array([1])
        data["sequence_lengths"][:] = t+1+1 
    return data

def cut_data_till_timestep(data, step, dummy_value):
    placeholder_covariates = dummy_value
    placeholder_outputs = dummy_value
    placeholder_treatments = np.array([1,0,0,0])
    placeholder_active_entries = np.array([0])
    data_cut = copy.deepcopy(data)
    for i in range(len(data['current_covariates'])):
        data_cut["prev_outputs"][i, step+1:, :]  = placeholder_covariates # current covariate sind um eins verschoben
        data_cut["current_covariates"][i, step+1:, :]  = placeholder_covariates # current covariate sind um eins verschoben
        # data_cut["outputs"][:,step+1:,:] = placeholder_outputs
        # data_cut["prev_treatments"][i, step+1+1+1:, :] = placeholder_treatments # previous treatments sind um eins verschoben und erster eintrag ist [0,0,0,0]
        # data_cut["current_treatments"][i, step+1+1:, :] = placeholder_treatments
        data_cut["active_entries"][i, step+1:] = placeholder_active_entries
        data_cut["sequence_lengths"][i] = step+1
    return data_cut

if __name__ == "__main__":
    main()

