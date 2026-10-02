#%%
import os
import pickle
import numpy as np
os.chdir('/Users/jaschob/Documents/GitHub/Promotion_Jennifer/Cancer_data_CF/Counterfactual_Datasets_For_ATE/Cancer/')

from src.utils.cancer_simulation import get_confounding_params, simulate, get_scaling_params
def generate_random_treatment_sequences(num_patients, num_time_steps):
    radio_choices = [0, 1, 2, 3]
    chemo_choices = [0, 3, 5, 7]
    radio_sequence = np.random.choice(radio_choices, size=num_time_steps)
    chemo_sequence = np.random.choice(chemo_choices, size=num_time_steps)

    assigned_actions = np.empty((num_patients, num_time_steps, 2), dtype=object)
    for i in range(num_patients):
        for t in range(num_time_steps):
            # Create one-hot probability vector for the chosen dose
            radio_idx = radio_choices.index(radio_sequence[t])
            radio_prob = [1.0 if j == radio_idx else 0.0 for j in range(4)]
            
            chemo_idx = chemo_choices.index(chemo_sequence[t])
            chemo_prob = [1.0 if j == chemo_idx else 0.0 for j in range(4)]
            
            assigned_actions[i, t, 0] = chemo_prob   # Chemo prob is index 0
            assigned_actions[i, t, 1] = radio_prob   # Radio prob is index 1

    return assigned_actions

def generate_no_treatment_sequences(num_patients, num_time_steps):
    assigned_actions = np.empty((num_patients, num_time_steps, 2), dtype=object)
    for i in range(num_patients):
        for t in range(num_time_steps):
            assigned_actions[i, t, 0] = [1.0, 0.0, 0.0, 0.0]  # 100% prob for dose 0 (Chemo)
            assigned_actions[i, t, 1] = [1.0, 0.0, 0.0, 0.0]  # 100% prob for dose 0 (Radio)
    return assigned_actions

def generate_only_max_radio_treatment_sequences(num_patients, num_time_steps):
    assigned_actions = np.empty((num_patients, num_time_steps, 2), dtype=object)
    for i in range(num_patients):
        for t in range(num_time_steps):
            assigned_actions[i, t, 0] = [1.0, 0.0, 0.0, 0.0]  # 100% prob for dose 0 (Chemo)
            assigned_actions[i, t, 1] = [0.0, 0.0, 0.0, 1.0]  # 100% prob for dose 3 (Radio)
    return assigned_actions

def generate_only_max_chemo_treatment_sequences(num_patients, num_time_steps):
    assigned_actions = np.empty((num_patients, num_time_steps, 2), dtype=object)
    for i in range(num_patients):
        for t in range(num_time_steps):
            assigned_actions[i, t, 0] = [0.0, 0.0, 0.0, 1.0]  # 100% prob for dose 7 (Chemo)
            assigned_actions[i, t, 1] = [1.0, 0.0, 0.0, 0.0]  # 100% prob for dose 0 (Radio)
    return assigned_actions


def generate_default_treatment_sequence(num_patients, num_time_steps, treatment_index=6):
    """
    Generate sequences where every time step uses the treatment corresponding to a single
    combined treatment index. Treatment indexing follows the nested ordering used elsewhere
    in the codebase: chemo choices vary slower and radio choices vary faster.

    chemo_choices = [0, 3, 5, 7]
    radio_choices = [0, 1, 2, 3]

    treatment_index -> chemo_idx = treatment_index // len(radio_choices)
                      -> radio_idx = treatment_index % len(radio_choices)

    The function returns assigned_actions shaped (num_patients, num_time_steps, 2)
    where axis 0 is chemo one-hot (len 4) and axis 1 is radio one-hot (len 4).
    """
    chemo_choices = [0, 3, 5, 7]
    radio_choices = [0, 1, 2, 3]

    num_chemo = len(chemo_choices)
    num_radio = len(radio_choices)
    max_index = num_chemo * num_radio - 1
    if treatment_index < 0 or treatment_index > max_index:
        raise ValueError(f"treatment_index must be between 0 and {max_index}")

    chemo_idx = treatment_index // num_radio
    radio_idx = treatment_index % num_radio

    assigned_actions = np.empty((num_patients, num_time_steps, 2), dtype=object)
    for i in range(num_patients):
        for t in range(num_time_steps):
            chemo_prob = [1.0 if j == chemo_idx else 0.0 for j in range(num_chemo)]
            radio_prob = [1.0 if j == radio_idx else 0.0 for j in range(num_radio)]
            assigned_actions[i, t, 0] = chemo_prob
            assigned_actions[i, t, 1] = radio_prob

    return assigned_actions

def generate_counterfactual_datasets():
    # Simulation Parameters
    seed = 100
    num_patients = 1000
    num_time_steps = 24
    window_size = 15
    chemo_coeff = 4
    radio_coeff = 4
    toxicity = True
    continuous_therapy = True
    intervention_start_t = 5
    
    # Get observational simulation params
    np.random.seed(seed)
    params = get_confounding_params(
        num_patients=num_patients,
        chemo_coeff=chemo_coeff,
        radio_coeff=radio_coeff,
        toxicity=toxicity
    )
    params["window_size"] = window_size
    
    # Save the random state so we can guarantee the SAME noise/randomness up to t=5
    random_state = np.random.get_state()
    
    # Define specific treatment sequences for dataset 1
    assigned_actions_1 = generate_random_treatment_sequences(num_patients, num_time_steps)
    assigned_actions_2 = generate_random_treatment_sequences(num_patients, num_time_steps)
    assigned_actions_3 = generate_only_max_radio_treatment_sequences(num_patients, num_time_steps)
    assigned_actions_4 = generate_only_max_chemo_treatment_sequences(num_patients, num_time_steps)
    assigned_actions_5 = generate_no_treatment_sequences(num_patients, num_time_steps)

    # Add default treatment as dataset 6 (map to combined index 6)
    # Note: treatment_index is in [0..(len(chemo_choices)*len(radio_choices)-1)]
    # Here we set default index to 6 as requested.
    assigned_actions_6 = generate_default_treatment_sequence(num_patients, num_time_steps, treatment_index=6)

    assigned_actions_list = [
        assigned_actions_1,
        assigned_actions_2,
        assigned_actions_3,
        assigned_actions_4,
        assigned_actions_5,
        assigned_actions_6,
    ]

    for idx, assigned_actions in enumerate(assigned_actions_list):
        # Generate Dataset
        print(f"Simulating Dataset {idx + 1}...")
        np.random.set_state(random_state)
        dataset = simulate(
            simulation_params=params,
            num_time_steps=num_time_steps,
            assigned_actions=assigned_actions,
            toxicity=toxicity,
            continuous_therapy=continuous_therapy,
            intervention_start_t=intervention_start_t
        )

        # Save datasets to pickle files
        os.makedirs("results", exist_ok=True)
        with open(f"results/dataset_{idx + 1}.p", "wb") as f:
            pickle.dump(dataset, f)
        
    print("Done! Generated five datasets that are identical up to t=5, then diverge based on treatment sequences.")

if __name__ == "__main__":
    generate_counterfactual_datasets()

# %%
