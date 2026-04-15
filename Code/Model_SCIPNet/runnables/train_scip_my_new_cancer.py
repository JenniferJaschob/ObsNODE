import logging
import hydra
import torch
import src.eval.my_vergleichsmasse as eval_script
import numpy as np
from pathlib import Path
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import AlphaRise, FilteringMlFlowLogger
# Ensure we use the improved time_varying_model_mod implementation when other modules
# import `src.models.time_varying_model` (e.g., `src.models.scip`). This avoids editing
# many files that import the default module. We insert the mod module into sys.modules
# under the original module name so subsequent imports resolve to the modified version.
import importlib
import sys
from src.models import time_varying_model
sys.modules['src.models.time_varying_model'] = time_varying_model

from src.models.scip import SCIP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for SCIP-Net
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    for seed in [106,107]:#[101,102,103,104,105]:
        # Initialisation of data to calculate dim_outcomes, dim_treatments, dim_vitals and dim_static_features
        seed_everything(seed)
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        assert args.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
        dataset_collection.process_data_encoder()
        args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
        args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
        args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
        args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

        # Train_callbacks
        prop_treatment_callbacks, propensity_history_callbacks, encoder_callbacks, decoder_callbacks = [], [], [], []

        # MlFlow Logger
        if args.exp.logging:
            experiment_name = f'{args.model.name}/{args.dataset.name}'
            mlf_logger = FilteringMlFlowLogger(filter_submodels=SCIP.possible_model_types, experiment_name=experiment_name,
                                            tracking_uri=args.exp.mlflow_uri)
            encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
            decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
            prop_treatment_callbacks += [LearningRateMonitor(logging_interval='epoch')]
            propensity_history_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        else:
            mlf_logger = None

        # ============================== Nominator (treatment propensity network) ==============================
        dataset_name = args.dataset.name
        save_dir = Path(get_original_cwd()) / 'model_save' / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"propensity_treatment_{seed}_cancer.ckpt"
        propensity_treatment = instantiate(args.model.propensity_treatment, args, dataset_collection, _recursive_=False)

        #if checkpoint_path.exists():
        #    logger.info(f'Loading propensity_treatment from {checkpoint_path}')
        #    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        #    propensity_treatment.load_state_dict(checkpoint['state_dict'])
        #    pt_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger)
        #    propensity_treatment.trainer = pt_trainer
        #else:
        if args.model.propensity_treatment.tune_hparams:
            propensity_treatment.finetune(resources_per_trial=args.model.propensity_treatment.resources_per_trial)

        propensity_treatment_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                                logger=mlf_logger,
                                                max_epochs=args.exp.max_epochs,
                                                callbacks=prop_treatment_callbacks,
                                                gradient_clip_val=args.model.propensity_treatment.max_grad_norm,
                                                terminate_on_nan=True)
        propensity_treatment_trainer.fit(propensity_treatment)
        propensity_treatment_trainer.save_checkpoint(str(checkpoint_path))

        # ============================== Denominator (history propensity network) ==============================
        checkpoint_path = save_dir / f"propensity_history_{seed}_cancer.ckpt"
        propensity_history = instantiate(args.model.propensity_history, args, dataset_collection, _recursive_=False)

        #if checkpoint_path.exists():
        #    logger.info(f'Loading propensity_history from {checkpoint_path}')
        #    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        #    propensity_history.load_state_dict(checkpoint['state_dict'])
        #    ph_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger)
        #    propensity_history.trainer = ph_trainer
        #else:
        if args.model.propensity_history.tune_hparams:
            propensity_history.finetune(resources_per_trial=args.model.propensity_history.resources_per_trial)

        propensity_history_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                                logger=mlf_logger,
                                                max_epochs=args.exp.max_epochs,
                                                callbacks=propensity_history_callbacks,
                                                gradient_clip_val=args.model.propensity_history.max_grad_norm,
                                                terminate_on_nan=True)
        propensity_history_trainer.fit(propensity_history)
        propensity_history_trainer.save_checkpoint(str(checkpoint_path))

        # ============================== Initialisation & Training of Encoder ==============================
        checkpoint_path = save_dir / f"encoder_{seed}_cancer.ckpt"
        encoder = instantiate(args.model.encoder, args, propensity_treatment, propensity_history, dataset_collection,
                                _recursive_=False)
        #if checkpoint_path.exists():
        #    logger.info(f'Loading encoder from {checkpoint_path}')
        #    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        #    encoder.load_state_dict(checkpoint['state_dict'])
        #    encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger)
        #    encoder.trainer = encoder_trainer
        #    encoder.prepare_data()
        #else:

        if args.model.encoder.tune_hparams:
            encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

        encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                    logger=mlf_logger,
                                    max_epochs=args.exp.max_epochs,
                                    callbacks=encoder_callbacks,
                                    gradient_clip_val=args.model.encoder.max_grad_norm,
                                    terminate_on_nan=True)
        encoder_trainer.fit(encoder)
        encoder_trainer.save_checkpoint(str(checkpoint_path))

        # ============================== Initialisation & Training of Decoder ==============================
        if args.model.train_decoder:
            checkpoint_path = save_dir / f"decoder_{seed}_cancer.ckpt"
            decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

            #if checkpoint_path.exists():
            #    logger.info(f'Loading decoder from {checkpoint_path}')
            #    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            #    decoder.load_state_dict(checkpoint['state_dict'])
            #    decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger)
            #    decoder.trainer = decoder_trainer
            #    decoder.prepare_data()
            #else:

            if args.model.decoder.tune_hparams:
                decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

            decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                        logger=mlf_logger,
                                        max_epochs=args.exp.max_epochs,
                                        gradient_clip_val=args.model.decoder.max_grad_norm,
                                        callbacks=decoder_callbacks,
                                        terminate_on_nan=True)
            decoder_trainer.fit(decoder)
            decoder_trainer.save_checkpoint(str(checkpoint_path))


        # ============================== Evaluation with Heatmaps ==============================

# ============================== Custom Evaluation ==============================
    logger.info("Setting up evaluation for heatmap...")
    
    x_pred_list = []
    my_run_trues = []
    list_index_t_s_pred = []
    
    test_f = dataset_collection.test_f
    max_seq_len = test_f.max_seq_length
    pure_test_data = deepcopy(test_f.data_original)
    
    device = torch.device('cpu')
    encoder.to(device)
    decoder.to(device)
    
    seeds = [106,107]#[101,102,103,104,105]
    
    for seed in seeds:
        logger.info(f"Evaluating Seed {seed}...")
        
        # Load the trained weights for this specific seed
        enc_ckpt = save_dir / f"encoder_{seed}_cancer.ckpt"
        dec_ckpt = save_dir / f"decoder_{seed}_cancer.ckpt"
        
        encoder.load_state_dict(torch.load(enc_ckpt, map_location=device, weights_only=False)['state_dict'])
        decoder.load_state_dict(torch.load(dec_ckpt, map_location=device, weights_only=False)['state_dict'])
        
        encoder.eval()
        decoder.eval()
        
        my_run_preds = []
        is_first_seed = (seed == seeds[0])
        
        # Loop over observation lengths
        for obs_len in range(1, max_seq_len):
            proj_horizon = max_seq_len - obs_len 
            if is_first_seed:
                logger.info(f"Evaluating: Observe {obs_len} steps, Predict {proj_horizon} steps ahead...")
            
            # Restore data and reset flags for this horizon
            test_f.data = deepcopy(pure_test_data)
            test_f.processed_sequential = False
            test_f.processed_autoregressive = False
            
            # Extract Encoder representations on test data using THIS seed's encoder
            r_test = encoder.get_representations(test_f)
            outputs_test = encoder.get_predictions(test_f)
            
            # Process test set for current projection horizon
            test_f.process_sequential_test(proj_horizon, r_test, save_encoder_r=False)
            test_f.process_autoregressive_test(r_test, outputs_test, proj_horizon, save_encoder_r=False)
            
            # Predict using THIS seed's decoder
            test_dataloader = DataLoader(test_f, batch_size=args.dataset.val_batch_size, shuffle=False)
            pred_list = []
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    pred = decoder(batch)
                    pred_list.append(pred.cpu())
                    
            preds = torch.cat(pred_list, dim=0)
            my_run_preds.append(preds)
            
            # We only need to extract and pad the ground truth ONCE, since the targets are always identical
            if is_first_seed:
                x_true_numpy = test_f.data_processed_seq['outputs']
                
                pad_len = obs_len - 1
                if pad_len > 0:
                    dummy_pad = np.zeros((x_true_numpy.shape[0], pad_len, x_true_numpy.shape[2]))
                    x_true_numpy = np.concatenate([dummy_pad, x_true_numpy], axis=1)
                    
                x_true_full = torch.from_numpy(x_true_numpy).to(device)
                my_run_trues.append(x_true_full)
                list_index_t_s_pred.append(obs_len - 1)
                
        x_pred_list.append(my_run_preds)
    
    logger.info("Computing custom heatmap metrics...")

    results_dir = Path(get_original_cwd()) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_script.heatmap_pred(
        x_pred_list=x_pred_list, 
        x_true=my_run_trues,      
        list_index_t_s_pred=list_index_t_s_pred,
        max_horizon=max_seq_len - 1,
        loss='rmse',
        save_link=Path(get_original_cwd()) / 'results' / 'rmse_heatmap_tumor.png',
        save_map=Path(get_original_cwd()) / 'results' / 'rmse_heatmap_tumor.pkl',
        index=0,
        vmin=0,
        vmax=1.0 
    )
    eval_script.heatmap_pred(
        x_pred_list=x_pred_list, 
        x_true=my_run_trues,      
        list_index_t_s_pred=list_index_t_s_pred,
        max_horizon=max_seq_len - 1,
        loss='rmse',
        save_link=Path(get_original_cwd()) / 'results' / 'rmse_heatmap_weight.png',
        save_map=Path(get_original_cwd()) / 'results' / 'rmse_heatmap_weight.pkl',
        index=1,
        vmin=0,
        vmax=1.0
    )
    return results


if __name__ == "__main__":
    main()

