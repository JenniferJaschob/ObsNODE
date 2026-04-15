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
        
        # Load and parse CSV (support both wide JSON-list per-row and long timepoint-per-row formats)
        df = pd.read_csv(file_path)
        self.data = {}

        # If values are JSON strings (wide format), parse them. Otherwise expect long format with subject_id/time.
        first_val = df['syn_outcome_1'].iloc[0]
        if isinstance(first_val, str):
            # wide format: each cell contains a JSON list
            self.data['syn_outcome_1'] = np.array([json.loads(x) for x in df['syn_outcome_1']])
            self.data['syn_treatment_1'] = np.array([json.loads(x) for x in df['syn_treatment_1']])
            self.data['syn_treatment_2'] = np.array([json.loads(x) for x in df['syn_treatment_2']])
            if 'sequence_lengths' in df.columns:
                self.data['sequence_lengths'] = df['sequence_lengths'].values
            else:
                self.data['sequence_lengths'] = np.array([arr.shape[0] for arr in self.data['syn_outcome_1']])
        else:
            # long format: rows are (subject_id, time, value). Group by subject_id and assemble trajectories.
            if 'subject_id' not in df.columns or 'time' not in df.columns:
                raise ValueError("Expected long-format CSV to contain 'subject_id' and 'time' columns when 'syn_outcome_1' is numeric")
            grouped = df.sort_values(['subject_id', 'time']).groupby('subject_id')
            syn_outcome_list = []
            t1_list = []
            t2_list = []
            seq_lengths = []
            for _, g in grouped:
                syn_outcome_list.append(g['syn_outcome_1'].values.astype(float))
                t1_list.append(g['syn_treatment_1'].values.astype(float))
                t2_list.append(g['syn_treatment_2'].values.astype(float))
                seq_lengths.append(len(g))
            max_len = max(seq_lengths) if len(seq_lengths) > 0 else 0
            def pad(arr_list):
                padded = np.full((len(arr_list), max_len), np.nan, dtype=float)
                for i, a in enumerate(arr_list):
                    padded[i, :len(a)] = a
                return padded
            self.data['syn_outcome_1'] = pad(syn_outcome_list)
            self.data['syn_treatment_1'] = pad(t1_list)
            self.data['syn_treatment_2'] = pad(t2_list)
            self.data['sequence_lengths'] = np.array(seq_lengths)

        self.num_patients, self.max_seq_length = self.data['syn_outcome_1'].shape
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
        # Collect outcome values up to each sequence length (ignore padded NaNs)
        outcomes = []
        for i in range(self.num_patients):
            l = int(self.data['sequence_lengths'][i])
            outcomes.extend(self.data['syn_outcome_1'][i, :l].tolist())

        outcomes = np.array(outcomes, dtype=float)
        means = np.array([np.mean(outcomes)])
        stds = np.array([np.std(outcomes)])
        max_val = np.array([np.max(outcomes)])

        return {
            'output_means': means,
            'output_stds': stds,
            'max_val': max_val
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
        norm_volume = (self.data['syn_outcome_1'] - means[0]) / stds[0]
        outcomes = norm_volume[:, :, np.newaxis]
        
        # pt_norm = (self.data['patient_types'] - scaling_params['patient_types_mean']) / scaling_params['patient_types_std']
        # patient_types_stacked = np.stack([pt_norm for t in range(self.max_seq_length)], axis=1)[:, :, np.newaxis]

        # Binarize treament applications (0: no treatment, 1: treatment)
        syn_treatment_1_binary = (self.data['syn_treatment_1'] > 0).astype(float)
        syn_treatment_2_binary = (self.data['syn_treatment_2'] > 0).astype(float)
        treatments = np.stack([syn_treatment_1_binary, syn_treatment_2_binary], axis=-1)
        
        treatment_times = (treatments.sum(axis=-1, keepdims=True) > 0) * 1.0

        # Model inputs/outputs (Shifted)
        self.data['outputs'] = outcomes[:, 1:, :]
        self.data['prev_outputs'] = outcomes[:, :-1, :]
        
        if self.treatment_mode == 'multiclass':
            # 0: None, 1: Chemo, 2: Radio, 3: Both
            one_hot = np.zeros((self.num_patients, self.max_seq_length, 4))
            for p in range(self.num_patients):
                for t in range(self.max_seq_length):
                    # use syn_treatment binaries for semisyn dataset
                    c, r = syn_treatment_1_binary[p, t], syn_treatment_2_binary[p, t]
                    if c == 0 and r == 0:
                        one_hot[p, t, 0] = 1
                    elif c > 0 and r == 0:
                        one_hot[p, t, 1] = 1
                    elif c == 0 and r > 0:
                        one_hot[p, t, 2] = 1
                    else:
                        one_hot[p, t, 3] = 1
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

                # If the available factual length is too small, we compute a safe start index
                # and copy only the available rows. Remaining entries stay zero (already initialized).
                # fact_length is the index of the first timestep used for prediction
                start_idx = fact_length - 1
                if start_idx < 0:
                    start_idx = 0

                # encoder init state (if available)
                if encoder_r is not None and (fact_length - 1) >= 0:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]

                # mark active encoder positions up to fact_length (if positive)
                if fact_length > 0:
                    seq2seq_active_encoder_r[i, :fact_length] = 1.0

                # Determine how many projection rows are actually available from source arrays
                avail_prev = max(0, previous_treatments.shape[1] - start_idx)
                avail_curr = max(0, current_treatments.shape[1] - (fact_length if fact_length >= 0 else 0))
                avail_out = max(0, outputs.shape[1] - (fact_length if fact_length >= 0 else 0))
                avail_cov = max(0, current_covariates.shape[1] - (fact_length if fact_length >= 0 else 0))

                # number of rows we can copy for projection horizon
                copy_prev = min(projection_horizon, avail_prev)
                copy_curr = min(projection_horizon, avail_curr)
                copy_out = min(projection_horizon, avail_out)
                copy_cov = min(projection_horizon, avail_cov)

                # Active entries: by default zeros; set available portion to ones
                if copy_out > 0:
                    seq2seq_active_entries[i, :copy_out, :] = np.ones((copy_out, 1))

                # Copy available slices into the seq2seq arrays
                if copy_prev > 0:
                    seq2seq_previous_treatments[i, :copy_prev, :] = previous_treatments[i, start_idx:start_idx + copy_prev, :]
                if copy_curr > 0:
                    seq2seq_current_treatments[i, :copy_curr, :] = current_treatments[i, max(0, fact_length):max(0, fact_length) + copy_curr, :]
                if copy_out > 0:
                    seq2seq_outputs[i, :copy_out, :] = outputs[i, max(0, fact_length):max(0, fact_length) + copy_out, :]
                    seq2seq_sequence_lengths[i] = copy_out
                else:
                    seq2seq_sequence_lengths[i] = 0

                # Disabled teacher forcing for test dataset: repeat last covariate if available
                if copy_cov > 0:
                    last_cov_idx = max(0, fact_length - 1)
                    seq2seq_current_covariates[i, :copy_cov, :] = np.repeat([current_covariates[i, last_cov_idx]], copy_cov, axis=0)
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
                # compute factual start index for prediction window
                fact_length = int(sequence_lengths[i]) - projection_horizon

                # safe encoder init if available
                if encoder_r is not None and (fact_length - 1) >= 0:
                    current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]

                # set current covariate init if available
                last_cov_idx = max(0, fact_length - 1)
                current_dataset['current_covariates'][i, 0, :dim_outcome] = encoder_outputs[i, last_cov_idx]

                if fact_length > 0:
                    current_dataset['active_encoder_r'][i, :fact_length] = 1.0

                # determine safe copy ranges (avoid broadcasting when shorter than projection_horizon)
                start_prev = max(0, fact_length - 1)
                start_curr = max(0, fact_length)

                avail_prev = max(0, prev_treatments.shape[1] - start_prev)
                avail_curr = max(0, current_treatments.shape[1] - start_curr)

                copy_prev = min(projection_horizon, avail_prev)
                copy_curr = min(projection_horizon, avail_curr)

                if copy_prev > 0:
                    current_dataset['prev_treatments'][i, :copy_prev, :] = prev_treatments[i, start_prev:start_prev + copy_prev, :]
                if copy_curr > 0:
                    current_dataset['current_treatments'][i, :copy_curr, :] = current_treatments[i, start_curr:start_curr + copy_curr, :]

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

def pad_or_truncate(arr_list, target=48):
    n = len(arr_list)
    out = np.full((n, target), np.nan, dtype=float)
    for i, a in enumerate(arr_list):
        L = min(len(a), target)
        out[i, :L] = a[:L]
    return out

class CustomDatasetCollection(SyntheticDatasetCollection):
    def __init__(self, train_path, val_path, test_path, projection_horizon=5, predict_weight=False, **kwargs):
        super().__init__()
        self.train_f = CustomCancerDataset(train_path, 'train', predict_weight=predict_weight)
        self.val_f = CustomCancerDataset(val_path, 'val', predict_weight=predict_weight)
        self.test_f = CustomCancerDataset(test_path, 'test', predict_weight=predict_weight)
        self.test_cf_one_step = self.test_f 
        self.test_cf_treatment_seq = self.test_f
        
        self.train_scaling_params = self.train_f.get_scaling_params()
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = False
