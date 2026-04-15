import pickle
import pandas as pd
import numpy as np
import json

def read_from_file(filename):
    return pickle.load(open(filename, "rb"))


def save_raw_dataset_to_csv(dataset_obj, filename_prefix):
    if dataset_obj is None:
        print(f"No dataset for {filename_prefix}, skipping...")
        return

    data = dataset_obj  
    entries = len(data['sequence_lengths']) 
    print(f"Number of entries: {entries}")
    all_rows = []

    for i in range(entries):
        row = {}

        for key in data.keys():
            entry = data[key][i]

            if isinstance(entry, (np.ndarray, list)):
                row[key] = json.dumps([float(x) for x in entry])

            else:
                row[key] = entry

        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    output_path = f"{filename_prefix}_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset {filename_prefix} saved to {output_path}")



def load_pickel_data_to_dict(path_x, path_u, timesteps=24):

    data_x = read_from_file(path_x)
    cancer_toxicity = data_x[1] 

    cancer_volume = cancer_toxicity[:timesteps, :, 0]  
    toxicity = cancer_toxicity[:timesteps, :, 1]  

    cancer_volume = cancer_volume.T 
    toxicity = toxicity.T 

    sequence_length = np.full(cancer_volume.shape[0], timesteps-1)

    data_treatment = read_from_file(path_u) 

    chemo_dosage = data_treatment[:timesteps, :, 0]  
    radio_dosage = data_treatment[:timesteps, :, 1] 

    chemo_dosage = chemo_dosage.T 
    radio_dosage = radio_dosage.T

    chemo_application = np.where(chemo_dosage > 7, 7, np.where(chemo_dosage > 5, 5, 0))
    radio_application = radio_dosage.numpy().copy() 
    print(sequence_length)

    result = {
        "cancer_volume": cancer_volume.numpy(),
        "toxicity": toxicity.numpy(),
        "sequence_lengths": sequence_length,
        "chemo_dosage": chemo_dosage.numpy(),
        "radio_dosage": radio_dosage.numpy(),
        "chemo_application": chemo_application,
        "radio_application": radio_application,
    }

    return result


if __name__ == "__main__":
    
    gamma = 20
    data_train_dict = load_pickel_data_to_dict("pickle_to_CSV/Dose_AI_cancer_data/data_train_cancer_DoseAI.pkl",
                                "pickle_to_CSV/Dose_AI_cancer_data/data_train_cancer_value_u_DoseAI.pkl", timesteps=24)
    data_val_dict = load_pickel_data_to_dict("pickle_to_CSV/Dose_AI_cancer_data/data_val_cancer_DoseAI.pkl",
                            "pickle_to_CSV/Dose_AI_cancer_data/data_val_cancer_value_u_DoseAI.pkl", timesteps=24)
    data_test_dict = load_pickel_data_to_dict(f"pickle_to_CSV/Dose_AI_cancer_data/data_test_cancer_gamma_{gamma}.pkl",
                                f"pickle_to_CSV/Dose_AI_cancer_data/data_test_cancer_value_u_gamma_{gamma}.pkl", timesteps=24)

    save_raw_dataset_to_csv(data_train_dict, "pickle_to_CSV/train_cancer_DoseAI")
    save_raw_dataset_to_csv(data_val_dict, "pickle_to_CSV/val_cancer_DoseAI")
    save_raw_dataset_to_csv(data_test_dict, f"pickle_to_CSV/test_cancer_gamma_{gamma}")
