import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import json
from copy import deepcopy

from src.data.dataset_collection import SyntheticDatasetCollection

logger = logging.getLogger(__name__)

class CustomCancerDataset(Dataset):
    """
    Adapter for custom CSV cancer data (DoseAI format)
    Can predict one outcome (tumor) or two outcomes (tumor + weight)
    """
    def __init__(self, file_path, subset_name, treatment_mode='multiclass', predict_weight=False):
        self.subset_name = subset_name
        self.treatment_mode = treatment_mode
        self.predict_weight = predict_weight
        
        # Load and parse CSV
        df = pd.read_csv(file_path)
        self.data = {}
        
        # Convert string lists to numpy arrays
        self.data['cancer_volume'] = np.array([json.loads(x) for x in df['cancer_volume']])
        self.data['toxicity'] = np.array([json.loads(x) for x in df['toxicity']]) 
        self.data['chemo_application'] = np.array([json.loads(x) for x in df['chemo_application']])
        self.data['radio_application'] = np.array([json.loads(x) for x in df['radio_application']])
        self.data['sequence_lengths'] = df['sequence_lengths'].values
        # self.data['patient_types'] = np.zeros(len(df))

        self.num_patients, self.max_seq_length = self.data['cancer_volume'].shape
        self.num_samples = self.num_patients
        
        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.norm_const = 13.0 

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items() if hasattr(v, '__len__') and len(v) == len(self)}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return self.data['current_covariates'].shape[0]

    def get_scaling_params(self):
        volumes = []
        tox_values = []
        for i in range(self.num_patients):
            l = int(self.data['sequence_lengths'][i])
            volumes.extend(self.data['cancer_volume'][i, :l])
            tox_values.extend(self.data['toxicity'][i, :l])

        if self.predict_weight:
            return {
                'output_means': np.array([np.mean(volumes), np.mean(tox_values)]),                                                  
                'output_stds': np.array([np.std(volumes), np.std(tox_values)]),
                # 'patient_types_mean': 0.0,
                # 'patient_types_std': 1.0,
                'max_val': np.array([np.max(volumes), np.max(tox_values)])
            }
        else:
            return {
                'output_means': np.array([np.mean(volumes)]),                                                                       
                'output_stds': np.array([np.std(volumes)]), 
                # 'patient_types_mean': 0.0,
                # 'patient_types_std': 1.0,
                'max_val': np.array([np.max(volumes)])
            }

    def process_data(self, scaling_params):
        if self.processed:
            return self.data
            
        logger.info(f'Processing {self.subset_name} dataset (predict_weight={self.predict_weight})')
        
        means = scaling_params['output_means']
        stds = scaling_params['output_stds']
        
        # Update norm_const to the max factual value found in training
        self.norm_const = scaling_params['max_val'][0] 
        
        # Normalize outcomes
        norm_volume = (self.data['cancer_volume'] - means[0]) / stds[0]
        if self.predict_weight:
            norm_toxicity = (self.data['toxicity'] - means[1]) / stds[1]
            outcomes = np.stack([norm_volume, norm_toxicity], axis=-1)
        else:
            outcomes = norm_volume[:, :, np.newaxis]
        
        # pt_norm = (self.data['patient_types'] - scaling_params['patient_types_mean']) / scaling_params['patient_types_std']
        # patient_types_stacked = np.stack([pt_norm for t in range(self.max_seq_length)], axis=1)[:, :, np.newaxis]

        # Binarize treament applications (0: no treatment, 1: treatment)
        chemo_binary = (self.data['chemo_application'] > 0).astype(float)
        radio_binary = (self.data['radio_application'] > 0).astype(float)
        treatments = np.stack([chemo_binary, radio_binary], axis=-1)
        
        treatment_times = (treatments.sum(axis=-1, keepdims=True) > 0) * 1.0

        # Model inputs/outputs (Shifted)
        self.data['outputs'] = outcomes[:, 1:, :]
        self.data['prev_outputs'] = outcomes[:, :-1, :]
        
        if self.treatment_mode == 'multiclass':
            # 0: None, 1: Chemo, 2: Radio, 3: Both
            one_hot = np.zeros((self.num_patients, self.max_seq_length, 4))
            for p in range(self.num_patients):
                for t in range(self.max_seq_length):
                    c, r = chemo_binary[p, t], radio_binary[p, t]
                    if c == 0 and r == 0: one_hot[p, t, 0] = 1
                    elif c > 0 and r == 0: one_hot[p, t, 1] = 1
                    elif c == 0 and r > 0: one_hot[p, t, 2] = 1
                    else: one_hot[p, t, 3] = 1
            self.data['current_treatments'] = one_hot[:, 1:, :]
            self.data['prev_treatments'] = one_hot[:, :-1, :]
        else:
            self.data['current_treatments'] = treatments[:, 1:, :]
            self.data['prev_treatments'] = treatments[:, :-1, :]

        self.data['current_treatment_times'] = treatment_times[:, 1:, :]
        self.data['prev_treatment_times'] = treatment_times[:, :-1, :]
        self.data['vitals'] = np.zeros((self.num_patients, self.max_seq_length - 1, 1))
        self.data['stabilized_weights'] = np.ones((self.num_patients, self.max_seq_length))
        # self.data['current_covariates'] = np.concatenate([self.data['prev_outputs'], patient_types_stacked[:, :-1, :]], axis=-1)
        self.data['current_covariates'] = self.data['prev_outputs']
        dim_outcome = outcomes.shape[-1]
        self.data['static_features'] = self.data['current_covariates'][:, 0, dim_outcome:]

        # Masking based on original lengths
        active_entries = np.zeros((self.num_patients, self.max_seq_length - 1, 1))
        for i in range(self.num_patients):
            l = int(self.data['sequence_lengths'][i])
            active_entries[i, :max(0, l-1), :] = 1
        self.data['active_entries'] = active_entries
        
        # Unscaled outputs for RMSE calculation
        self.data['unscaled_outputs'] = self.data['outputs'] * stds + means
        
        self.scaling_params = scaling_params
        self.processed = True
        return self.data

    def process_sequential(self, encoder_r, projection_horizon, save_encoder_r=False):
        """
        Pre-process dataset for multiple-step-ahead prediction: explodes dataset to a larger one with rolling origin
        Args:
            encoder_r: Representations of encoder
            projection_horizon: Projection horizon
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before training (multiple sequences)')

            outputs = self.data['outputs']
            sequence_lengths = self.data['sequence_lengths']
            active_entries = self.data['active_entries']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments'][:, 1:, :]  # Without zero_init_treatment
            current_covariates = self.data['current_covariates']
            stabilized_weights = self.data['stabilized_weights'] if 'stabilized_weights' in self.data else None

            num_patients, seq_length, num_features = outputs.shape

            num_seq2seq_rows = num_patients * seq_length

            seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, seq_length))
            seq2seq_original_index = np.zeros((num_seq2seq_rows, ))
            seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
            seq2seq_current_covariates = np.zeros((num_seq2seq_rows, projection_horizon, current_covariates.shape[-1]))
            seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
            seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, projection_horizon + 1)) \
                if stabilized_weights is not None else None

            total_seq2seq_rows = 0  # we use this to shorten any trajectories later

            for i in range(num_patients):

                sequence_length = int(sequence_lengths[i])

                for t in range(1, sequence_length - projection_horizon):  # shift outputs back by 1
                    seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[i, t - 1, :]  # previous state output
                    seq2seq_original_index[total_seq2seq_rows] = i
                    seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                    max_projection = min(projection_horizon, sequence_length - t)

                    seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[i, t:t + max_projection, :]
                    seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = \
                        previous_treatments[i, t - 1:t + max_projection - 1, :]
                    seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = \
                        current_treatments[i, t:t + max_projection, :]
                    seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t:t + max_projection, :]
                    seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                    seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :] = \
                        current_covariates[i, t:t + max_projection, :]

                    if seq2seq_stabilized_weights is not None:  # Also including SW of one-step-ahead prediction
                        seq2seq_stabilized_weights[total_seq2seq_rows, :] = stabilized_weights[i, t - 1:t + max_projection]

                    total_seq2seq_rows += 1

            # Filter everything shorter
            seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
            seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]
            seq2seq_active_encoder_r = seq2seq_active_encoder_r[:total_seq2seq_rows, :]
            seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_covariates = seq2seq_current_covariates[:total_seq2seq_rows, :, :]
            seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
            seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
            seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
            if seq2seq_stabilized_weights is not None:
                seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

            # Package outputs
            dim_outcome = seq2seq_outputs.shape[-1]
            seq2seq_data = {
                'init_state': seq2seq_state_inits,
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'current_covariates': seq2seq_current_covariates,
                'prev_outputs': seq2seq_current_covariates[:, :, :dim_outcome],
                'static_features': seq2seq_current_covariates[:, 0, dim_outcome:],
                'outputs': seq2seq_outputs,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            }
            if seq2seq_stabilized_weights is not None:
                seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :seq_length, :]

            self.processed_sequential = True
            self.exploded = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: takes the last n-steps according to the projection horizon
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before testing (multiple sequences)')

            sequence_lengths = self.data['sequence_lengths']
            outputs = self.data['outputs']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments'][:, 1:, :]  # Without zero_init_treatment
            current_covariates = self.data['current_covariates']

            num_patient_points, max_seq_length, num_features = outputs.shape

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
            seq2seq_current_covariates = np.zeros((num_patient_points, projection_horizon, current_covariates.shape[-1]))
            seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
            seq2seq_sequence_lengths = np.zeros(num_patient_points)

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                seq2seq_previous_treatments[i] = previous_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_outputs[i] = outputs[i, fact_length: fact_length + projection_horizon, :]
                seq2seq_sequence_lengths[i] = projection_horizon
                # Disabled teacher forcing for test dataset
                seq2seq_current_covariates[i] = np.repeat([current_covariates[i, fact_length - 1]], projection_horizon, axis=0)
            dim_outcome = outputs.shape[-1]
            # Package outputs
            seq2seq_data = {
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'current_covariates': seq2seq_current_covariates,
                'prev_outputs': seq2seq_current_covariates[:, :, :dim_outcome],
                'static_features': seq2seq_current_covariates[:, 0, dim_outcome:],
                'outputs': seq2seq_outputs,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
                'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
                # 'patient_types': self.data['patient_types'],
                # 'patient_ids_all_trajectories': self.data['patient_ids_all_trajectories'],
                # 'patient_current_t': self.data['patient_current_t']
            }
            if encoder_r is not None:
                seq2seq_data['init_state'] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: axillary dataset placeholder for autoregressive prediction
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            logger.info(f'Processing {self.subset_name} dataset before testing (autoregressive)')

            current_treatments = self.data_original['current_treatments']
            prev_treatments = self.data_original['prev_treatments'][:, 1:, :]  # Without zero_init_treatment

            sequence_lengths = self.data_original['sequence_lengths']
            num_patient_points, max_seq_length = current_treatments.shape[:2]

            current_dataset = dict()  # Same as original, but only with last n-steps
            current_dataset['current_covariates'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_covariates'].shape[-1]))
            current_dataset['prev_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                           self.data_original['prev_treatments'].shape[-1]))
            current_dataset['current_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_treatments'].shape[-1]))
            current_dataset['init_state'] = np.zeros((num_patient_points, encoder_r.shape[-1]))
            current_dataset['active_encoder_r'] = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            current_dataset['active_entries'] = np.ones((num_patient_points, projection_horizon, 1))

            dim_outcome = encoder_outputs.shape[-1]
            
            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
                current_dataset['current_covariates'][i, 0, :dim_outcome] = encoder_outputs[i, fact_length - 1]
                current_dataset['active_encoder_r'][i, :fact_length] = 1.0
                current_dataset['prev_treatments'][i] = \
                    prev_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                current_dataset['current_treatments'][i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]

            current_dataset['prev_outputs'] = current_dataset['current_covariates'][:, :, :dim_outcome]
            current_dataset['static_features'] = self.data_original['static_features']

            self.data_processed_seq = deepcopy(self.data)
            self.data = current_dataset
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data

class CustomDatasetCollection(SyntheticDatasetCollection):
    def __init__(self, train_path, val_path, test_path, train_path_static=None, val_path_static=None, test_path_static=None, projection_horizon=5, predict_weight=False, treatment_mode='multiclass', **kwargs):
        super().__init__()
        # CustomCancerDataset expects (file_path, subset_name, treatment_mode=..., predict_weight=...)
        # The config typically provides only train/val/test paths, so static paths are optional and unused here.
        self.train_f = CustomCancerDataset(train_path, 'train', treatment_mode=treatment_mode, predict_weight=predict_weight)
        self.val_f = CustomCancerDataset(val_path, 'val', treatment_mode=treatment_mode, predict_weight=predict_weight)
        self.test_f = CustomCancerDataset(test_path, 'test', treatment_mode=treatment_mode, predict_weight=predict_weight)

        self.test_cf_one_step = self.test_f 
        self.test_cf_treatment_seq = self.test_f
        
        self.train_scaling_params = self.train_f.get_scaling_params()
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False