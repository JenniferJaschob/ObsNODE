import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import json
from copy import deepcopy
import pickle

from src.data.dataset_collection import SyntheticDatasetCollection

logger = logging.getLogger(__name__)



class CustomCancerDataset(Dataset):

    def __init__(self, file_path, subset_name, treatment_mode='multiclass', predict_weight=False,
                 already_standardized: bool = False):
        self.subset_name = subset_name
        self.treatment_mode = treatment_mode
        self.predict_weight = predict_weight
        self.already_standardized = already_standardized
        
        with open(file_path, 'rb') as handle:
            df = pickle.load(handle)
        df_dynamic = df['dynamic']
        df_static = df['static']

        self.data = {}
        first_val = df_dynamic['SOFA'].iloc[0]
        if isinstance(first_val, str):
            dynamic_value = ['SOFA', 'creatinine', 'bilirubin_total', 'alt',
                'index', 'aniongap', 'bicarbonate', 'bun', 'dbp', 'platelet', 'rdw',
                'sbp', 'SOFA_mask', 'aniongap_mask', 'bicarbonate_mask', 'bun_mask',
                'creatinine_mask', 'dbp_mask', 'platelet_mask', 'rdw_mask', 'sbp_mask',
                'bilirubin_total_mask', 'alt_mask', 'Vancomycin',
                'Piperacillin-Tazobactam', 'Ceftriaxon']

            static_value = ['subject_id', 'admission_age', 'height', 'weight', 'male',
                'admission_age_mask', 'height_mask', 'weight_mask', 'male_mask',
                'Vancomycinstat', 'Piperacillin-Tazobactamstat', 'Ceftriaxonstat']    

            for val in dynamic_value:
                self.data[val] = np.array([json.loads(x) for x in df[val]])
            for val in static_value:
                self.data[val] = np.array(df[val])    
            # sequence_lengths may be stored at the top-level of the pickled dict or as a
            # DataFrame column. Handle both cases safely.
            if isinstance(df, dict) and 'sequence_lengths' in df:
                self.data['sequence_lengths'] = np.array(df['sequence_lengths'])
            elif hasattr(df, 'columns') and 'sequence_lengths' in df.columns:
                self.data['sequence_lengths'] = df['sequence_lengths'].values
            else:
                # infer sequence lengths from the first dynamic channel
                first_dyn = dynamic_value[0]
                if first_dyn in self.data and len(self.data[first_dyn]) > 0:
                    # if stored as 2D array, get second dim; if stored as list-of-lists, infer lengths
                    arr0 = self.data[first_dyn]
                    if isinstance(arr0, np.ndarray) and arr0.ndim == 2:
                        self.data['sequence_lengths'] = np.array([int(np.count_nonzero(~np.isnan(arr0[i, :]))) for i in range(arr0.shape[0])])
                    else:
                        self.data['sequence_lengths'] = np.array([len(arr) for arr in arr0])
                else:
                    self.data['sequence_lengths'] = np.zeros((len(df,),), dtype=int)

            # Optionally interpolate/backfill outcome channels (SOFA, creatinine, bilirubin_total, alt)
            try:
                # Ensure outcome arrays are numeric 2D arrays; if so, interpolate NaNs along time
                for outcome_col in ['SOFA', 'creatinine', 'bilirubin_total', 'alt']:
                    if outcome_col in self.data and isinstance(self.data[outcome_col], np.ndarray) and self.data[outcome_col].ndim == 2:
                        if np.isnan(self.data[outcome_col]).any():
                            # use linear interpolation along time and bfill/ffill at edges
                            self.data[outcome_col] = interpolate_and_bfill_array(self.data[outcome_col])
            except Exception:
                # keep original arrays if interpolation fails
                pass
        else:
            # long format: rows are (subject_id, time, value). Group by subject_id and assemble trajectories.
            # The loaded object may be a top-level DataFrame or a dict containing a 'dynamic'
            # DataFrame. Prefer the top-level if it is a DataFrame, otherwise use df_dynamic.
            df_for_cols = df if hasattr(df, 'columns') else (df_dynamic if hasattr(df_dynamic, 'columns') else None)
            if df_for_cols is None or 'subject_id' not in df_for_cols.columns or 'time' not in df_for_cols.columns:
                raise ValueError("Expected long-format CSV to contain 'subject_id' and 'time' columns when 'SOFA' is numeric")
            grouped = df_for_cols.sort_values(['subject_id', 'time']).groupby('subject_id')
            outcome1_list = []
            outcome2_list = []
            outcome3_list = []
            outcome4_list = []
            t1_list = []
            t2_list = []
            t3_list = []
            seq_lengths = []
            for _, g in grouped:
                outcome1_list.append(g['SOFA'].values.astype(float))
                outcome2_list.append(g['creatinine'].values.astype(float))
                outcome3_list.append(g['bilirubin_total'].values.astype(float))
                outcome4_list.append(g['alt'].values.astype(float))
                t1_list.append(g['Vancomycin'].values.astype(float))
                t2_list.append(g['Piperacillin-Tazobactam'].values.astype(float))
                t3_list.append(g['Ceftriaxon'].values.astype(float))
                seq_lengths.append(len(g))
            max_len = max(seq_lengths) if len(seq_lengths) > 0 else 0
            def pad(arr_list):
                padded = np.full((len(arr_list), max_len), np.nan, dtype=float)
                for i, a in enumerate(arr_list):
                    padded[i, :len(a)] = a
                return padded
            self.data['SOFA'] = pad(outcome1_list)
            self.data['creatinine'] = pad(outcome2_list)
            self.data['bilirubin_total'] = pad(outcome3_list)
            self.data['alt'] = pad(outcome4_list)
            self.data['Vancomycin'] = pad(t1_list)
            self.data['Piperacillin-Tazobactam'] = pad(t2_list)
            self.data['Ceftriaxon'] = pad(t3_list)
            self.data['sequence_lengths'] = np.array(seq_lengths)

        self.num_patients, self.max_seq_length = self.data['SOFA'].shape
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
        SOFA = []
        creatinine = []
        bilirubin_total = []
        alt = []
        for i in range(self.num_patients):
            l = int(self.data['sequence_lengths'][i])
            SOFA.extend(self.data['SOFA'][i, :l])
            creatinine.extend(self.data['creatinine'][i, :l])
            bilirubin_total.extend(self.data['bilirubin_total'][i, :l])
            alt.extend(self.data['alt'][i, :l])

        return {
                'output_means': np.array([np.nanmean(SOFA), np.nanmean(creatinine), np.nanmean(bilirubin_total), np.nanmean(alt)]),                                                  
                'output_stds': np.array([np.nanstd(SOFA), np.nanstd(creatinine), np.nanstd(bilirubin_total), np.nanstd(alt)]),
                # 'patient_types_mean': 0.0,
                # 'patient_types_std': 1.0,
                'max_val': np.array([np.max(SOFA), np.max(creatinine), np.max(bilirubin_total), np.max(alt)])
            }
  

    def process_data(self, scaling_params):
        if self.processed:
            return self.data

        logger.info(f'Processing {self.subset_name} dataset (predict_weight={self.predict_weight})')
    

        means = scaling_params['output_means']
        stds = scaling_params['output_stds']
        
        # Update norm_const to the max factual value found in training
        self.norm_const = scaling_params['max_val'][0] 
        
        
        # pt_norm = (self.data['patient_types'] - scaling_params['patient_types_mean']) / scaling_params['patient_types_std']
        # patient_types_stacked = np.stack([pt_norm for t in range(self.max_seq_length)], axis=1)[:, :, np.newaxis]

        # Binarize treament applications (0: no treatment, 1: treatment)
        treatment_1_binary = (self.data['Vancomycin'] > 0).astype(float)
        treatment_2_binary = (self.data['Piperacillin-Tazobactam'] > 0).astype(float)
        treatment_3_binary = (self.data['Ceftriaxon'] > 0).astype(float)
        treatments = np.stack([treatment_1_binary, treatment_2_binary, treatment_3_binary], axis=-1)
        
        treatment_times = (treatments.sum(axis=-1, keepdims=True) > 0) * 1.0

        # Stack outcome channels into final tensor
        #sofa = (self.data['SOFA'] - means[0]) / stds[0]
        #creatinine = (self.data['creatinine'] - means[1]) / stds[1]
        #bilirubin_total = (self.data['bilirubin_total'] - means[2]) / stds[2]
        #alt = (self.data['alt'] - means[3]) / stds[3]
        #outcomes = np.stack([sofa, creatinine, bilirubin_total, alt], axis=-1)
        outcomes = np.stack([self.data['SOFA'], self.data['creatinine'], self.data['bilirubin_total'], self.data['alt']], axis=-1)
        ## stacked shape: (num_patients, max_seq_length, 4)
        #outcomes = stacked 
        #obs_mask_raw = (~np.isnan(stacked)).astype(float)
        obs_mask_raw = (~np.isnan(outcomes)).astype(float)

        # Model inputs/outputs (Shifted)
        self.data['outputs'] = outcomes[:, 1:, :]
        self.data['prev_outputs'] = outcomes[:, :-1, :]
        
        # Forward-fill NaNs in normalized space; use fill_value=0.0 (entspricht Feature-mean)
        #self.data['prev_outputs'] = ffill(self.data['prev_outputs'], fill_value=0.0)
        #self.data['outputs'] = ffill(self.data['outputs'], fill_value=0.0)

        if self.treatment_mode == 'multiclass':
            # 0: None, 1: Chemo, 2: Radio, 3: Both
            one_hot = np.zeros((self.num_patients, self.max_seq_length, 4))
            for p in range(self.num_patients):
                for t in range(self.max_seq_length):
                    # use syn_treatment binaries for semisyn dataset
                    c, r, s = treatment_1_binary[p, t], treatment_2_binary[p, t], treatment_3_binary[p, t]
                    if c == 0 and r == 0 and s == 0:
                        one_hot[p, t, 0] = 1
                    elif c > 0 and r == 0 and s == 0:
                        one_hot[p, t, 1] = 1
                    elif c == 0 and r > 0 and s == 0:
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
        # If vitals were present and contain NaNs, interpolate linearly along time and backfill/forward-fill
        try:
            if 'vitals' in self.data and self.data['vitals'] is not None:
                # apply interpolation/backfill per patient/channel when NaNs are present
                if np.isnan(self.data['vitals']).any():
                    self.data['vitals'] = interpolate_and_bfill_array(self.data['vitals'])
        except Exception:
            # keep original vitals if interpolation fails
            pass
        self.data['stabilized_weights'] = np.ones((self.num_patients, self.max_seq_length))
        
        static_cols = ['subject_id', 'admission_age', 'height', 'weight', 'male',
                'admission_age_mask', 'height_mask', 'weight_mask', 'male_mask',
                'Vancomycinstat', 'Piperacillin-Tazobactamstat', 'Ceftriaxonstat']    

        patient_static_list = []
        for col in static_cols:
            if col in self.data:
                patient_static_list.append(np.asarray(self.data[col], dtype=float))
            else:
                patient_static_list.append(np.zeros((self.num_patients,), dtype=float))

        # Stack into shape (N, D_static)
        patient_static = np.stack(patient_static_list, axis=-1)

        # Tile static features along time axis so they can be concatenated to prev_outputs
        patient_types_stacked = np.repeat(patient_static[:, np.newaxis, :], self.max_seq_length, axis=1)  # (N, T, D_static)
        self.data['current_covariates'] = np.concatenate([self.data['prev_outputs'], patient_types_stacked[:, :-1, :]], axis=-1)
        self.data['static_features'] = patient_static #self.data['current_covariates'][:, 0, dim_outcome:]
        
        
        #self.data['current_covariates'] = self.data['prev_outputs']
        #self.data['static_features'] = self.data['current_covariates'][:, 0, dim_outcome:]
        
        dim_outcome = outcomes.shape[-1]
        # Masking based on original lengths
        active_entries = np.zeros((self.num_patients, self.max_seq_length - 1, 1))
        for i in range(self.num_patients):
            l = int(self.data['sequence_lengths'][i])
            active_entries[i, :max(0, l-1), :] = 1
        self.data['active_entries'] = active_entries
        # Store observation mask for later use (mask corresponds to original stacked, shifted to prev/cur)
        # prev_obs_mask: mask for prev_outputs (aligned with prev_outputs shape)
        try:
            self.data['obs_mask'] = obs_mask_raw
            self.data['obs_mask_prev'] = obs_mask_raw[:, :-1, :]
        except Exception:
            # if something unexpected happens, ignore but keep processing
            pass

        # Unscaled outputs for RMSE calculation
        # create unscaled outputs (useful when RMSE should be computed in original scale)
        self.data['unscaled_outputs'] = self.data['outputs'] #* stds + means

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
                'unscaled_outputs': seq2seq_outputs, #* self.scaling_params['output_stds'] + self.scaling_params['output_means'],
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
                'unscaled_outputs': seq2seq_outputs,# * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
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
        # Note: older configs may provide *_path_static entries; the current
        # CustomCancerDataset implementation reads dynamics (and optional
        # statics) from the single CSV `train_path`. Ignore static path keys
        # here and instantiate with the expected signature.
        self.train_f = CustomCancerDataset(train_path, 'train', predict_weight=predict_weight)
        self.val_f = CustomCancerDataset(val_path, 'val', predict_weight=predict_weight)
        self.test_f = CustomCancerDataset(test_path, 'test', predict_weight=predict_weight)
        self.test_cf_one_step = self.test_f 
        self.test_cf_treatment_seq = self.test_f
        
        self.train_scaling_params = self.train_f.get_scaling_params()
        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = True


import numpy as np

def _ffill_2d(arr2d, fill_value=0.0):
    # arr2d: shape (B, T)
    arr = arr2d.copy()
    # Replace leading NaNs in first column with fill_value
    mask_first = np.isnan(arr[:, 0])
    if mask_first.any():
        arr[mask_first, 0] = fill_value
    # boolean mask of valid entries
    mask = np.isnan(arr)
    # indices of last non-nan per row
    idx = np.where(~mask, np.arange(arr.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out

def ffill(arr, fill_value=0.0):
    """
    Forward-fill NaNs along time axis (axis=1).
    Accepts arrays shaped (B, T) or (B, T, D).
    Leading NaNs are replaced with `fill_value`.
    """
    if arr is None:
        return arr
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return _ffill_2d(arr, fill_value=fill_value)
    elif arr.ndim == 3:
        B, T, D = arr.shape
        out = np.empty_like(arr)
        for d in range(D):
            out[:, :, d] = _ffill_2d(arr[:, :, d], fill_value=fill_value)
        return out
    else:
        raise ValueError("ffill supports only 2D (B,T) or 3D (B,T,D) arrays")


def interpolate_and_bfill_array(arr):
    """Interpolate NaNs along time axis (axis=1) for numpy arrays.
    Supports shapes (B, T) and (B, T, D). Uses pandas interpolation per-row.
    Missing values at the ends are backfilled/forward-filled.
    """
    if arr is None:
        return arr
    arr = np.asarray(arr)
    if arr.ndim == 2:
        B, T = arr.shape
        out = np.copy(arr)
        for i in range(B):
            s = pd.Series(out[i, :])
            s = s.interpolate(method='linear', limit_direction='both')
            s = s.fillna(method='bfill').fillna(method='ffill')
            out[i, :] = s.values
        return out
    elif arr.ndim == 3:
        B, T, D = arr.shape
        out = np.copy(arr)
        for d in range(D):
            # build DataFrame of shape (B, T) for this channel and interpolate per-row
            df = pd.DataFrame(out[:, :, d])
            df = df.interpolate(method='linear', axis=1, limit_direction='both')
            df = df.fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
            out[:, :, d] = df.values
        return out
    else:
        # unsupported shape
        return arr


###

#all_vitals = all_vitals.interpolate(method='linear')
#all_vitals = all_vitals.fillna(method='bfill')