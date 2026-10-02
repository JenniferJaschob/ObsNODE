import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_datasets(ds1_path="results/dataset_1.p", ds2_path="results/dataset_2.p", num_samples=5):
    # Load the datasets
    if not os.path.exists(ds1_path) or not os.path.exists(ds2_path):
        print(f"Error: Could not find {ds1_path} or {ds2_path}.")
        print("Make sure you run generate_counterfactual_datasets.py first!")
        return

    with open(ds1_path, "rb") as f:
        ds1 = pickle.load(f)
    with open(ds2_path, "rb") as f:
        ds2 = pickle.load(f)

    fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(15, 3 * num_samples))
    fig.suptitle("Comparison of Counterfactual Datasets (First {} Samples)".format(num_samples), fontsize=16)

    # Plot for each of the first 'num_samples' patients
    for i in range(num_samples):
        seq_len_1 = int(ds1["sequence_lengths"][i])
        seq_len_2 = int(ds2["sequence_lengths"][i])
        
        # We'll plot up to the maximum sequence length of the two for this patient
        max_t = max(seq_len_1, seq_len_2)
        t_axis = np.arange(max_t)

        # 1) Cancer Volume
        ax_vol = axes[i, 0]
        ax_vol.plot(t_axis[:seq_len_1], ds1["cancer_volume"][i, :seq_len_1], label="Dataset 1", marker='o', markersize=3)
        ax_vol.plot(t_axis[:seq_len_2], ds2["cancer_volume"][i, :seq_len_2], label="Dataset 2", marker='x', markersize=3, linestyle='--')
        ax_vol.axvline(x=5, color='red', linestyle=':', label='Intervention (t=5)')
        ax_vol.set_ylabel(f"Patient {i}\nCancer Volume")
        if i == 0:
            ax_vol.set_title("Tumor Trajectory")
            ax_vol.legend()
        if i == num_samples - 1:
            ax_vol.set_xlabel("Time Step")

        # 2) Chemo Application
        ax_chemo = axes[i, 1]
        ax_chemo.step(t_axis[:seq_len_1], ds1["chemo_application"][i, :seq_len_1], label="Dataset 1", where='post')
        ax_chemo.step(t_axis[:seq_len_2], ds2["chemo_application"][i, :seq_len_2], label="Dataset 2", where='post', linestyle='--')
        ax_chemo.axvline(x=5, color='red', linestyle=':')
        ax_chemo.set_ylabel("Chemo Dose")
        if i == 0:
            ax_chemo.set_title("Chemo Application")
            ax_chemo.legend()
        if i == num_samples - 1:
            ax_chemo.set_xlabel("Time Step")

        # 3) Radio Application
        ax_radio = axes[i, 2]
        ax_radio.step(t_axis[:seq_len_1], ds1["radio_application"][i, :seq_len_1], label="Dataset 1", where='post')
        ax_radio.step(t_axis[:seq_len_2], ds2["radio_application"][i, :seq_len_2], label="Dataset 2", where='post', linestyle='--')
        ax_radio.axvline(x=5, color='red', linestyle=':')
        ax_radio.set_ylabel("Radio Dose")
        if i == 0:
            ax_radio.set_title("Radio Application")
            ax_radio.legend()
        if i == num_samples - 1:
            ax_radio.set_xlabel("Time Step")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    out_file = "results/dataset_comparison.png"
    plt.savefig(out_file)
    print(f"Successfully generated comparison plot and saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    compare_datasets()
