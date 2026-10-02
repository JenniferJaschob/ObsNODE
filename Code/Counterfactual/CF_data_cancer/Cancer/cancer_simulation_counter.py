"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018 &
https://github.com/ioanabica/Counterfactual-Recurrent-Network

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

Notes:
- Simulation time taken to be in days

"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import (
    truncnorm,  # we need to sample from truncated normal distributions
)
import itertools
#from src.utils.data_utils import write_to_file, process_data
#from data_utils import write_to_file, process_data

#from src.utils.data_utils import process_data
from data_utils import process_data

from tqdm import tqdm #new causal transformer


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL: https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed
def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0


# Tumour constants per

# Tumour constants per
tumour_cell_density = 5.8 * 10.0**8.0  # cells per cm^3
tumour_death_threshold = calc_volume(13)  # assume spherical

TUMOUR_CELL_DENSITY = tumour_cell_density #new causal transformer
TUMOUR_DEATH_THRESHOLD = tumour_death_threshold #new causal transformer

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist

tumour_size_distributions = {
    #"I": (1.72, 4.70, 0.3, 5.0),
    #"II": (1.96, 1.63, 0.3, 13.0),
    #"IIIA": (1.91, 9.40, 0.3, 13.0),
    #"IIIB": (2.76, 6.87, 0.3, 13.0),
    "IV": (7, 4.82, 4.5, 13.0),
    #'IV': (3.86, 8.82, 0.3, 13.0)
}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {
    #"I": 1432,
    #"II": 128,
    #"IIIA": 1306,
    #"IIIB": 7248,
    "IV": 12840,
}



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions


def get_confounding_params(num_patients, chemo_coeff, radio_coeff, toxicity=False):
    """

    Get original simulation parameters, and add extra ones to control confounding

    :param num_patients:
    :param chemo_coeff: Bias on action policy for chemotherapy assignments
    :param radio_activation_group: Bias on action policy for chemotherapy assignments
    :return:
    """

    basic_params = get_standard_params(num_patients)
    patient_types = basic_params["patient_types"]
    tumour_stage_centres = [s for s in cancer_stage_observations if "IIIA" not in s]
    tumour_stage_centres.sort()

    d_max = calc_diameter(tumour_death_threshold)
    basic_params["chemo_sigmoid_intercepts"] = np.array([d_max / 2.0 for i in patient_types],)
    basic_params["radio_sigmoid_intercepts"] = np.array([d_max / 2.0 for i in patient_types],)

    basic_params["chemo_sigmoid_betas"] = np.array([chemo_coeff / d_max for i in patient_types],
    )
    basic_params["radio_sigmoid_betas"] = np.array([radio_coeff / d_max for i in patient_types],)
    
    if toxicity:
        #initial weight ist nötig
        #Vorschlag: dwnet/dt = kqWnet(t) -kl1 A1(t) -kl2 A2(t) -kl3 T(t)
        #body mass
        kg = np.array(
            #[7*0.00002 for i in patient_types])
            [np.random.normal(0.00014,0.00001) for i in patient_types])
        #chemotherapy
        kl1 = np.array(
            #[2.25*0.00035 for i in patient_types])
            #[1.25*0.00142 for i in patient_types])
            [np.random.normal(0.001775,0.0001) for i in patient_types])
        
        #radiotherapy
        kl2 = np.array(
            #[2.25*0.0075 for i in patient_types])
            #[1.25*0.0033 for i in patient_types])
            [np.random.normal(0.004125,0.0002) for i in patient_types])
        #colon cancer effect
        kl3 = np.array(
            #[2.25*0.0000253 for i in patient_types])
            #[1.25*0.0000253 for i in patient_types])
            [np.random.normal(0.000031625,0.000015) for i in patient_types])
        
        #kl1_mean_adjustments = np.array([0.025 if i > 1 else 0 for i in patient_types])
        #kl2_mean_adjustments = np.array([0.025 if i < 3 else 0 for i in patient_types])
        #kl3_mean_adjustments = np.array([0.00001 for i in patient_types])
        
        kl1_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
        kl2_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])
        kl3_mean_adjustments = np.array([0.0 if i != 2 else 0.1 for i in patient_types])
        
        # norm_rvs1 = truncnorm.rvs(
        #     0,
        #     2,
        #     size=patient_types.shape[0],
        # )
        # norm_rvs2 = truncnorm.rvs(
        #     0,
        #     2,
        #     size=patient_types.shape[0],
        # )
        # norm_rvs3 = truncnorm.rvs(
        #     0,
        #     2,
        #     size=patient_types.shape[0],
        # )
        # norm_rvskg = truncnorm.rvs(
        #     0,
        #     2,
        #     size=patient_types.shape[0],
        # )
        
        basic_params['kl1'] = kl1 + kl1 * kl1_mean_adjustments#*norm_rvs1
        basic_params['kl2'] = kl2 + kl2 * kl2_mean_adjustments#*norm_rvs2
        basic_params['kl3'] = kl3 + kl3 * kl3_mean_adjustments#*norm_rvs3
        basic_params['kg'] = kg# + kg * kg_mean_adjustments*norm_rvskg
        
        #Initialgewicht?!
        basic_params['initial_toxicity']  = np.array(
            [np.random.lognormal(mean=np.log(100),sigma=0.2) for i in patient_types])
        
        
        
    return basic_params

def get_standard_params(num_patients):  # additional params
    """
    Simulation parameters from the Nature article + adjustments for static variables

    :param num_patients:
    :return: simulation_parameters
    """

    # Adjustments for static variables
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types, num_patients)
    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])
    
    total = 0
    for k in cancer_stage_observations:
        total += cancer_stage_observations[k]
    cancer_stage_proportions = {
        k: float(cancer_stage_observations[k]) / float(total)
        for k in cancer_stage_observations
    }
    
    # INITIAL VOLUMES SAMPLING
    #TOTAL_OBS = sum(cancer_stage_observations.values())
    #cancer_stage_proportions = {k: cancer_stage_observations[k] / TOTAL_OBS for k in cancer_stage_observations}


    # remove possible entries
    possible_stages = list(tumour_size_distributions.keys())
    possible_stages.sort()
   # possible_stages=possible_stages


    initial_stages = np.random.choice(
        possible_stages,
        num_patients,
        p=[cancer_stage_proportions[k] for k in possible_stages],
    )

    # Get info on patient stages and initial volumes
    output_initial_diam = []
    patient_sim_stages = []
    for stg in possible_stages:
        count = np.sum((initial_stages == stg) * 1)

        mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

        # Convert lognorm bounds in to standard normal bounds
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma

        logging.info(
            (
                "Simulating initial volumes for stage {} "
                + " with norm params: mu={}, sigma={}, lb={}, ub={}"
            ).format(stg, mu, sigma, lower_bound, upper_bound),
        )

        norm_rvs = truncnorm.rvs(
            lower_bound,
            upper_bound,
            size=count,
        )  # truncated normal for realistic clinical outcome

        initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_volume_by_stage)
        patient_sim_stages += [stg for i in range(count)]

    # Fixed params
    K = calc_volume(30)  # carrying capacity given in cm, so convert to volume
    alpha_beta_ratio = 10
    alpha_rho_corr = 0.87

    # Distributional parameters for dynamics
    parameter_lower_bound = 0.0
    parameter_upper_bound = 0.3 #Grenze angepasst
    rho_params = (7 * 10**-5, 7.23 * 10**-3)
    alpha_params = (0.0398, 0.088)
    beta_c_params = (0.02, 0.0007)

    # Get correlated simulation paramters (alpha, beta, rho) which respects bounds
    alpha_rho_cov = np.array(
        [
            [alpha_params[1] ** 2, alpha_rho_corr * alpha_params[1] * rho_params[1]],
            [alpha_rho_corr * alpha_params[1] * rho_params[1], rho_params[1] ** 2],
        ],
    )

    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

    simulated_params = []

    while (
        len(simulated_params) < num_patients
    ):  # Keep on simulating till we get the right number of params

        param_holder = np.random.multivariate_normal(
            alpha_rho_mean,
            alpha_rho_cov,
            size=num_patients,
        )

        for i in range(param_holder.shape[0]):

            # Ensure that all params fulfill conditions
            if (
                param_holder[i, 0] > parameter_lower_bound
                and param_holder[i, 1] > parameter_lower_bound
                and param_holder[i, 0] < parameter_upper_bound
                and param_holder[i, 1] < parameter_upper_bound
            ):
                simulated_params.append(param_holder[i, :])

        logging.info(
            "Got correlated params for {} patients".format(len(simulated_params)),
        )

    simulated_params = np.array(simulated_params)[
        :num_patients, :
    ]  # shorten this back to normal
    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0]*0.75 + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / alpha_beta_ratio
    
    # Get the remaining indep params
    logging.info("Simulating beta c parameters")
    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = (
        beta_c_params[0]*0.75
        + beta_c_params[1]
        * truncnorm.rvs(
            (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
            (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
            size=num_patients,
        )
        + beta_c_adjustments
    )

    output_holder = {
        "patient_types": patient_types,
        "initial_stages": np.array(patient_sim_stages),
        #here changed!!!!
        #"initial_volumes": calc_volume(
        #    np.array(output_initial_diam),
        #),  # assumed spherical with diam
        "initial_volumes": calc_volume(
            np.array(output_initial_diam),
        ),  # assumed spherical with diam
        "alpha": alpha,
        "rho": rho,
        "beta": beta,
        "beta_c": beta_c,
        "K": np.array([K for i in range(num_patients)]),
    }
    # np.random.exponential(expected_treatment_delay, num_patients),

    # Randomise output params
    logging.info("Randomising outputs")
    idx = [i for i in range(num_patients)]
    np.random.shuffle(idx)

    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]
    
    return output_params

#used from get_cancer_sim_data
def simulate(simulation_params, num_time_steps, assigned_actions=None, toxicity=False, continuous_therapy=False):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]
    
    if continuous_therapy:
        chemo_amt=[3,5,7]
        radio_amt=[1,2,3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params["initial_stages"]
    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params[
        "window_size"
    ]  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]
    
    

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))#sb
    radio_dosage = np.zeros((num_patients, num_time_steps))#sb
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    chemo_probabilities = np.zeros((num_patients, num_time_steps))#sb
    radio_probabilities = np.zeros((num_patients, num_time_steps))#sb

    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, num_time_steps))
        noise_terms_toxic = 0.0015 * np.random.randn(
            num_patients,
            num_time_steps,
        )

    noise_terms = 0.01 * np.random.randn(
        num_patients,
        num_time_steps,
    )  # 5% cell variability
    
    recovery_rvs = np.random.rand(num_patients, num_time_steps) #sb

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)#sb
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)#sb
    
    if continuous_therapy:
        chemo_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, num_time_steps])
        radio_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, num_time_steps])
        #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7

    # Run actual simulation
    for i in range(num_patients):
        if i % 200 == 0:
            logging.info("Simulating patient {} of {}".format(i, num_patients))
        noise = noise_terms[i]
        

        # initial values
        # factual values counterfactual case (heare sb)
        # factual_cancer_volume = np.zeros(num_time_steps)
        # factual_chemo_dosage = np.zeros(num_time_steps)
        # factual_radio_dosage = np.zeros(num_time_steps)
        # factual_chemo_application_point = np.zeros(num_time_steps)
        # factual_radio_application_point = np.zeros(num_time_steps)
        # factual_chemo_probabilities = np.zeros(num_time_steps)
        # factual_radio_probabilities = np.zeros(num_time_steps)
        # chemo_application_rvs = np.random.rand(num_time_steps)
        # radio_application_rvs = np.random.rand(num_time_steps)
        
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]

        for t in range(0, num_time_steps - 1):

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used],).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:

                radio_prob = 1.0 / (1.0+ np.exp(-radio_sigmoid_betas[i]* (cancer_metric_used - radio_sigmoid_intercepts[i]),))
                chemo_prob = 1.0 / (1.0+ np.exp(-chemo_sigmoid_betas[i]* (cancer_metric_used - chemo_sigmoid_intercepts[i]),))
            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob:
                radio_application_point[i, t] = 1
                if continuous_therapy:
                    if radio_prob+radio_dosage_rvs[i,t]<0.25:
                        radio_dosage[i, t] = radio_amt[0]
                    elif radio_prob+radio_dosage_rvs[i,t]>0.25 and radio_prob+radio_dosage_rvs[i,t]<0.5:
                        radio_dosage[i, t] = radio_amt[1]
                    else:
                        radio_dosage[i, t] = radio_amt[2]
                    radio_application_point[i, t] = radio_dosage[i,t]
                        
                else:
                    radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:
                chemo_application_point[i, t] = 1
                #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
                if continuous_therapy:
                    if chemo_prob+chemo_dosage_rvs[i,t]<0.25:
                        current_chemo_dose = chemo_amt[0]
                    elif chemo_prob+chemo_dosage_rvs[i,t]>0.25 and chemo_prob+chemo_dosage_rvs[i,t]<0.5:
                        current_chemo_dose = chemo_amt[1]
                    else:
                        current_chemo_dose = chemo_amt[2]
                    chemo_application_point[i, t] = current_chemo_dose
                        
                else:
                    current_chemo_dose = chemo_amt[0]
                    
            # Update chemo dosage
            
            
            # use in counterfactual case (other index?)
            # factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # # Factual treatments and outcomes #same as chemo_volume (different nois index)
            # factual_cancer_volume[t + 1] = factual_cancer_volume[t] * (1 + rho * np.log(K / factual_cancer_volume[t])
            #                                                            - beta_c * factual_chemo_dosage[t] - (
            #                                                                    alpha * factual_radio_dosage[
            #                                                                t] + beta * factual_radio_dosage[t] ** 2)
            #                                                            + noise[t + 1])  # add noise to fit residuals

            # factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, tumour_death_threshold)

            
            chemo_dosage[i, t] = (
                previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
                + current_chemo_dose
            )

            cancer_volume[i, t + 1] = cancer_volume[i, t] * (
                1
                + rho * np.log(K / cancer_volume[i, t])
                - beta_c * chemo_dosage[i, t]
                - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
                + noise[t]
            )  # add noise to fit residuals
            
            # if i==0:
            #     if t==0:
            #         print(alpha)
            #         print(beta)
            #     print('t')
            #     print(rho * np.log(K / cancer_volume[i, t])) #ok, kein Problem
            #     print(beta_c * chemo_dosage[i, t]) #ok
            #     print(alpha * radio_dosage[i, t])
            #     print( beta * radio_dosage[i, t] ** 2)
            #     print(noise[t])
            #     print(cancer_volume[i, t + 1])
            #     print((
            #         1
            #         + rho * np.log(K / cancer_volume[i, t])
            #         - beta_c * chemo_dosage[i, t]
            #         - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
            #         + noise[t]
            #     ))
                
            if cancer_volume[i, t + 1] > tumour_death_threshold:
                #print(i)
                cancer_volume[i, t + 1] = tumour_death_threshold
                break  # patient death
            
            # if cancer_volume[i,t+1]<0:
            #     print(i)
            #     print(alpha)
            
            if recovery_rvs[i, t + 1]/50 > cancer_volume[i, t + 1]: #np.exp(
                #-cancer_volume[i, t + 1] * tumour_cell_density,
            #):
                cancer_volume[i, t + 1] = 0
                break
            
            if toxicity:
                toxic[i, t + 1] = toxic[i, t] * (
                    1
                    + kg * toxic[i, t] *(1-toxic[i,t]/(initial_toxicity[i])) 
                    - kl1 * chemo_dosage[i, t]
                    - kl2 * radio_dosage[i, t]
                    - kl3 * cancer_volume[i, t]
                    + noise_toxic[t]
                )


        # Package outputs
        sequence_lengths[i] = int(t + 1)
    
    if toxicity:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": chemo_dosage,
            "radio_dosage": radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": chemo_probabilities,
            "radio_probabilities": radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
            "toxicity": toxic
        }
        
    else:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": chemo_dosage,
            "radio_dosage": radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": chemo_probabilities,
            "radio_probabilities": radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
        }

    return outputs

# # Wie funktion alsol ...
# ergänzende Kommentare aus causal transformer Paper
# def simulate_test(num_time_steps, chemo_options=[0,3,5,7],radio_options=[0,1,2,3], num_patients=10, chemo_coeff=4, radio_coeff=4, window_size=15, toxicity=True, seed=200):
#     """
#     Core routine to generate simulation paths

#     :param simulation_params:
#     :param num_time_steps:
#     :param assigned_actions:
#     :return:
#     """

#     np.random.seed(seed)
    
#     simulation_params = get_confounding_params(
#         int(num_patients),
#         chemo_coeff=chemo_coeff,
#         radio_coeff=radio_coeff,
#         toxicity=toxicity
#     )
#     simulation_params["window_size"] = window_size
    
#     test_data_factuals = simulate(simulation_params, num_time_steps+1,toxicity=toxicity,continuous_therapy=True)
    
#     total_num_chemo_treatments = 1


#     chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

#     # sort this
#     chemo_idx = np.argsort(chemo_days)
#     chemo_days = np.array(chemo_days)[chemo_idx]
    
#     chemo_amt=chemo_options
#     radio_amt=radio_options
#     treatment_options=list(itertools.product(chemo_amt,radio_amt))

#     drug_half_life = 1  # one day half life for drugs

#     # Unpack simulation parameters
#     initial_stages = simulation_params["initial_stages"]
#     initial_volumes = simulation_params["initial_volumes"]
#     alphas = simulation_params["alpha"]
#     rhos = simulation_params["rho"]
#     betas = simulation_params["beta"]
#     beta_cs = simulation_params["beta_c"]
#     Ks = simulation_params["K"]

#     num_patients = initial_stages.shape[0]
    
#     #extrem viele Optionen, vermutlich müssen wir das kürzen
#     #das geht nichtmal mit 10 Zeitschritten.... entweder weniger
#     #Zeitschritte oder weniger Behandlungen. Mit 10 Patienten geht das
#     number_options = (len(chemo_amt)*len(radio_amt))**num_time_steps
    
#     #patients*options*time_steps
#     cancer_volume = np.zeros((num_patients, number_options, num_time_steps+1))
#     cancer_volume[:]=np.nan
#     chemo_dosage = np.zeros((num_patients, number_options, num_time_steps))
#     chemo_dosage[:]=np.nan
#     radio_dosage = np.zeros((num_patients, number_options, num_time_steps))
#     radio_dosage[:]=np.nan
#     chemo_application_point = np.zeros((num_patients, number_options, num_time_steps))
#     chemo_application_point[:]=np.nan
#     radio_application_point = np.zeros((num_patients, number_options, num_time_steps))
#     radio_application_point[:]=np.nan
#     sequence_lengths = np.zeros((num_patients, number_options))
#     sequence_lengths[:]=np.nan
    
#     if toxicity:
#         initial_toxicity = simulation_params["initial_toxicity"]
#         kgs = simulation_params["kg"]
#         kl1s = simulation_params["kl1"]
#         kl2s = simulation_params["kl2"]
#         kl3s = simulation_params["kl3"]
#         toxic = np.zeros((num_patients, number_options, num_time_steps+1))
#         toxic[:]=np.nan
#         noise_terms_toxic = 0.0015 * np.random.randn(
#             num_patients,
#             num_time_steps,
#         )

#     noise_terms = 0.01 * np.random.randn(
#         num_patients,
#         num_time_steps,
#     )  # 5% cell variability
    
#     recovery_rvs = np.random.rand(num_patients, num_time_steps+1)
    
#     for i in range(num_patients):
#         #if i % 200 == 0:
#         logging.info("Simulating patient {} of {}".format(i, num_patients))
#         noise = noise_terms[i]        

#         # initial values
#         cancer_volume[i, :, 0] = initial_volumes[i]
#         alpha = alphas[i]
#         beta = betas[i]
#         beta_c = beta_cs[i]
#         rho = rhos[i]
#         K = Ks[i]
#         if toxicity:
#             noise_toxic = noise_terms_toxic[i]
#             toxic[i,:, 0] = initial_toxicity[i]
#             kg=kgs[i]
#             kl1=kl1s[i]
#             kl2=kl2s[i]
#             kl3=kl3s[i]
        
#         for t in range(0, num_time_steps):
#             #outerloop for number of indices
#             index=int(number_options/((len(radio_amt)*len(chemo_amt))**(t+1)))
#             for j in range((len(radio_amt)*len(chemo_amt))**(t+1)):
#                 treatment_index=j % ((len(radio_amt)*len(chemo_amt)))
#                 #print(treatment_index)
#                 #chem = treatment_options[j][0]
#                 #radio = treatment_options[j][1]
#                 current_chemo_dose = 0.0
#                 previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, j*index, t - 1]
                
#                 # Action application
#                 radio_application_point[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][1]
#                 radio_dosage[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][1]

#                 current_chemo_dose = treatment_options[treatment_index][0]
#                 chemo_application_point[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][0]
#                 # Update chemo dosage
#                 chemo_dosage[i, j*index:(j+1)*index, t] = (
#                     previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
#                     + current_chemo_dose
#                 )
    
#                 cancer_volume[i, j*index:(j+1)*index, t + 1] = cancer_volume[i, j*index:(j+1)*index, t] * (
#                     1
#                     + rho * np.log(K / cancer_volume[i, j*index:(j+1)*index, t])
#                     - beta_c * chemo_dosage[i, j*index:(j+1)*index, t]
#                     - (alpha * radio_dosage[i, j*index:(j+1)*index, t] + beta * radio_dosage[i, j*index:(j+1)*index, t] ** 2)
#                     + noise[t]
#                 )  # add noise to fit residuals
                    
#                 if toxicity:
#                     toxic[i, j*index:(j+1)*index, t + 1] = toxic[i, j*index:(j+1)*index, t] * (
#                         1
#                         + kg * toxic[i, j*index:(j+1)*index, t] *(1-toxic[i, j*index:(j+1)*index,t]/(initial_toxicity[i])) 
#                         - kl1 * chemo_dosage[i, j*index:(j+1)*index, t]
#                         - kl2 * radio_dosage[i, j*index:(j+1)*index, t]
#                         - kl3 * cancer_volume[i, j*index:(j+1)*index, t]
#                         + noise_toxic[t]
#                     )
                
#                 if cancer_volume[i, j*index, t + 1] > tumour_death_threshold:
#                     #print(i)
                    
#                     cancer_volume[i, j*index:(j+1)*index, t + 1] = tumour_death_threshold
#                     break  # patient death
    
#                 #if recovery_rvs[i, t + 1]/50 > cancer_volume[i, j*index, t + 1]: #np.exp(
#                 #    #-cancer_volume[i, t + 1] * tumour_cell_density,
#                 #):
#                 #    cancer_volume[i, j*index:(j+1)*index, t + 1] = 0
#                 #    break
                
#                 sequence_lengths[i, j*index:(j+1)*index] = int(t + 1)

#     #cancer_volume=np.delete(cancer_volume,pd.isna(cancer_volume[:,0]),axis=0)
#     #chemo_dosage=np.delete(chemo_dosage,pd.isna(cancer_volume[:,0]),axis=0)
#     #radio_dosage=np.delete(radio_dosage,pd.isna(cancer_volume[:,0]),axis=0)
#     #chemo_application_point=np.delete(chemo_application_point,pd.isna(cancer_volume[:,0]),axis=0)
#     #radio_application_point=np.delete(radio_application_point,pd.isna(cancer_volume[:,0]),axis=0)
#     #sequence_lengths=np.delete(sequence_lengths,pd.isna(sequence_lengths),axis=0)
    
#     outputs = {
#         "cancer_volume": cancer_volume,
#         #"chemo_dosage": chemo_dosage,
#         #"radio_dosage": radio_dosage,
#         "chemo_application": chemo_application_point,
#         "radio_application": radio_application_point,
#         "sequence_lengths": sequence_lengths,
#         "toxicity": toxic
#     }

#     return outputs, test_data_factuals

def get_scaling_params(sim, toxicity=False,continuous=False):
    real_idx = ["cancer_volume", "chemo_dosage", "radio_dosage"]
    if toxicity:
        real_idx.append("toxicity")
    if continuous:
        real_idx.append("chemo_application")
        real_idx.append("radio_application")

    # df = pd.DataFrame({k: sim[k] for k in real_idx})
    means = {}
    stds = {}
    seq_lengths = sim["sequence_lengths"]
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means["patient_types"] = np.mean(sim["patient_types"])
    stds["patient_types"] = np.std(sim["patient_types"])

    return pd.Series(means), pd.Series(stds)

# from Philipps Cancer Paper
def get_cancer_sim_data(
    chemo_coeff,
    radio_coeff,
    b_load,
    b_save=False,
    seed=100,
    model_root="results",
    window_size=15,
    num_time_steps=60,
    num_patients=10000,
    max_horizon=5,
    toxicity=False,
    num_patients_test=1000,
    continuous_therapy=False, 
    assigned_actions=None,
    counterfactual=False,
    projection_horizon=5,
    cf_seq_mode='sliding_treatment'
):
    if window_size == 15:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}.p".format(chemo_coeff, radio_coeff),
        )
    else:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}_{}.p".format(chemo_coeff, radio_coeff, window_size),
        )

    def _generate():
        np.random.seed(seed)

        #train
        params_train = get_confounding_params(num_patients,chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
        params_train["window_size"] = window_size
        training_data = simulate(params_train, num_time_steps, toxicity=toxicity,continuous_therapy=continuous_therapy,assigned_actions=assigned_actions)

        #val
        params_val = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
        params_val["window_size"] = window_size
        validation_data = simulate(params_val, num_time_steps,toxicity=toxicity,continuous_therapy=continuous_therapy,assigned_actions=assigned_actions)

        #test
        params_test = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
        params_test["window_size"] = window_size
        test_data_factuals = simulate(params_test, num_time_steps,toxicity=toxicity,continuous_therapy=continuous_therapy,assigned_actions=assigned_actions)
        
        if counterfactual == True:
            #conterfaftual test one step
            params_test_1 = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
            params_test_1["window_size"] = window_size
            test_data_counterfactual_1_step = simulate_counterfactual_1_step(simulation_params=params_test_1, seq_length=num_time_steps,toxicity=False,continuous_therapy=continuous_therapy)#simulation_params, seq_length, toxicity=False,continuous_therapy = False
            
            #conterfaftual test seq
            params_test_seq = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
            params_test_seq["window_size"] = window_size
            test_data_counterfactual_seq =  simulate_counterfactuals_treatment_seq(simulation_params=params_test_seq, seq_length=num_time_steps, projection_horizon=projection_horizon, cf_seq_mode=cf_seq_mode,toxicity=False,continuous_therapy=continuous_therapy)
        #validation_data=None
        #test_data_factuals=None
            #test counterfactual
            params_test_counter = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
            params_test_counter["window_size"] = window_size
            test_data_counterfactual_my1 = simulate_with_counterfactuals_my1(params_test_counter, num_time_steps, assigned_actions=None,toxicity=True, continuous_therapy=True)


            params_test_counter_myseq = get_confounding_params(int(num_patients_test),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
            params_test_counter_myseq["window_size"] = window_size
            counterfactual_seqs = [np.array([[3.0, 1.0]] * 10),  # z.B. 10 Tage mit 3 Gy Chemo, 1 Gy Radio
                                    np.array([[5.0, 2.0]] * 10),]  # 10 Tage mit 5 Gy Chemo, 2 Gy Radio
                                
            test_data_counterfactual_myseq = simulate_with_counterfactuals_myseq(
                                        params_test_counter_myseq,
                                        num_time_steps,
                                        assigned_actions=None,
                                        toxicity=True,
                                        continuous_therapy=True,
                                        counterfactual_treatment_sequences=counterfactual_seqs,  # Neu
                                        projection_horizon=5,                      # Neu
                                        counterfactual_start_times=[10]            # Neu
                                    )
        params = get_confounding_params(int(num_patients),chemo_coeff=chemo_coeff,radio_coeff=radio_coeff,toxicity=toxicity)
        params["window_size"] = window_size

        scaling_data = get_scaling_params(training_data, toxicity=toxicity,continuous=continuous_therapy)
        

        
        if counterfactual == True:
            pickle_map = {
                "chemo_coeff": chemo_coeff,
                "radio_coeff": radio_coeff,
                "num_time_steps": num_time_steps,
                "training_data": training_data,
                "validation_data": validation_data,
                "test_data_factuals": test_data_factuals,
                "scaling_data": scaling_data,
                "window_size": window_size,
                "test_data_counterfactual_1_step": test_data_counterfactual_1_step,
                "test_data_counterfactual_seq": test_data_counterfactual_seq,
                "test_data_counterfactual_my1": test_data_counterfactual_my1,
                "test_data_counterfactual_myseq": test_data_counterfactual_myseq,
                
            }
        else:
            pickle_map = {
                "chemo_coeff": chemo_coeff,
                "radio_coeff": radio_coeff,
                "num_time_steps": num_time_steps,
                "training_data": training_data,
                "validation_data": validation_data,
                "test_data_factuals": test_data_factuals,
                "scaling_data": scaling_data,
                "window_size": window_size
            }
               
        if b_save:
            logging.info("Saving pickle map to {}".format(pickle_file))
            pickle.dump(pickle_map, open(pickle_file, "wb"))
        return pickle_map

    # Controls whether to regenerate the data, or load from a persisted file
    if not b_load:
        pickle_map = _generate()

    else:
        logging.info("Loading pickle map from {}".format(pickle_file))

        try:
            pickle_map = pickle.load(open(pickle_file, "rb"))

        except IOError:
            logging.info(
                "Pickle file does not exist, regenerating: {}".format(pickle_file),
            )
            pickle_map = _generate()

    return pickle_map

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fuer eine Behandlungsaenderungsoptionsset (Chemo und Radio Kombie)
#rufe iterativ diese Funktion auf um konterfaktische behandlungen zu simulieren... speichere vorher nicht alle Kombinationen ab...
# bei mehreren Counterfaktischen überprüfungen gehe iterativ über diese funktion
# diese fuktion muss iterativ bis zu einem Schritt aufgerufen werden, aktualisiert bei der entsprechenden Behandlung
# 
#bei transformer Paper haben die versucht eine eigene Funktion dazu geschrieben

# def simulate_true_model(time, output_earlier=None, initial=False, scaling_data=None, max_time=10, chemo=0,radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200):
#     #output_earlier describes the output of the model of one earlier iteration
#     #initial true initializes the first iteration
    
#     """
#     Core routine to generate simulation paths

#     :param simulation_params:
#     :param num_time_steps:
#     :param assigned_actions:
#     :return:
#     """

#     np.random.seed(seed)
     
#     # No earlier data
#     if initial:
#         simulation_params = get_confounding_params(
#             int(num_patients),
#             chemo_coeff=chemo_coeff,
#             radio_coeff=radio_coeff,
#             toxicity=True
#         )
#         simulation_params["window_size"] = window_size  
#         noise_terms_toxic = 0.0015 * np.random.randn(
#             num_patients,
#             max_time,
#         )
    
#         noise_terms = 0.01 * np.random.randn(
#             num_patients,
#             max_time,
#         )
#         simulation_params["noise_terms"]=noise_terms
#         simulation_params["noise_terms_toxic"]=noise_terms_toxic
#     else:
#         simulation_params=output_earlier["simulation_params"]
#         noise_terms=simulation_params["noise_terms"]
#         noise_terms_toxic=simulation_params["noise_terms_toxic"]

#     total_num_chemo_treatments = 1

#     chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

#     # sort this
#     chemo_idx = np.argsort(chemo_days)
#     chemo_days = np.array(chemo_days)[chemo_idx]

#     drug_half_life = 1  # one day half life for drugs

#     # Unpack simulation parameters
#     initial_stages = simulation_params["initial_stages"]
#     initial_volumes = simulation_params["initial_volumes"]
#     initial_toxicity = simulation_params["initial_toxicity"]
    
#     alphas = simulation_params["alpha"]
#     rhos = simulation_params["rho"]
#     betas = simulation_params["beta"]
#     beta_cs = simulation_params["beta_c"]
#     Ks = simulation_params["K"]
#     kgs = simulation_params["kg"]
#     kl1s = simulation_params["kl1"]
#     kl2s = simulation_params["kl2"]
#     kl3s = simulation_params["kl3"]
#     patient_types = simulation_params["patient_types"]
    
#     if initial:
#         cancer_volume = np.zeros((num_patients, max_time+1))
#         cancer_volume[:] = np.nan
#         chemo_dosage = np.zeros((num_patients, max_time+1))
#         chemo_dosage[:] = np.nan
#         radio_dosage = np.zeros((num_patients, max_time+1))
#         radio_dosage[:] = np.nan
#         chemo_application_point = np.zeros((num_patients, max_time+1))
#         chemo_application_point[:] = np.nan
#         radio_application_point = np.zeros((num_patients, max_time+1))
#         radio_application_point[:] = np.nan
#         sequence_lengths = np.zeros((num_patients))
#         toxic = np.zeros((num_patients, max_time+1))
#         toxic[:] = np.nan
#     else:
#         cancer_volume = output_earlier["cancer_volume"]
#         chemo_dosage = output_earlier["chemo_dosage"]
#         radio_dosage = output_earlier["radio_dosage"]
#         chemo_application_point = output_earlier["chemo_application"]
#         radio_application_point = output_earlier["radio_application"]
#         sequence_lengths = output_earlier["sequence_lengths"]
#         toxic = output_earlier["toxicity"]
    
#     for i in range(num_patients):
#         noise = noise_terms[i]        
        
#         # initial values
#         if initial:
#             cancer_volume[i, 0] = initial_volumes[i]
#             toxic[i, 0] = initial_toxicity[i]
#             sequence_lengths[i] = max_time

#         else:
#             alpha = alphas[i]
#             beta = betas[i]
#             beta_c = beta_cs[i]
#             rho = rhos[i]
#             K = Ks[i]
#             noise_toxic = noise_terms_toxic[i]
#             kg=kgs[i]
#             kl1=kl1s[i]
#             kl2=kl2s[i]
#             kl3=kl3s[i]
            
#             previous_chemo_dose = 0.0 if time == 0 else chemo_dosage[i, time - 1]
            
#             # Action application
#             radio_application_point[i, time] = radio
#             radio_dosage[i, time] = radio

#             current_chemo_dose = chemo
#             chemo_application_point[i, time] = chemo
#             # Update chemo dosage
#             chemo_dosage[i, time] = (
#                 previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
#                 + current_chemo_dose
#             )
            
#             if cancer_volume[i, time]>0:
    
#                 cancer_volume[i, time + 1] = cancer_volume[i, time] * (
#                     1
#                     + rho * np.log(K / cancer_volume[i, time])
#                     - beta_c * chemo_dosage[i, time]
#                     - (alpha * radio_dosage[i, time] + beta * radio_dosage[i, time] ** 2)
#                     + noise[time]
#                 )
                
#                 toxic[i, time + 1] = toxic[i, time] * (
#                     1
#                     + kg * toxic[i, time] *(1-toxic[i,time]/(initial_toxicity[i])) 
#                     - kl1 * chemo_dosage[i, time]
#                     - kl2 * radio_dosage[i, time]
#                     - kl3 * cancer_volume[i, time]
#                     + noise_toxic[time]
#                 )
#             else:
#                 cancer_volume[i, time + 1] = 0
#                 toxic[i, time + 1] = 0
                
#             tumour_death_threshold=calc_volume(13)
            
#             if cancer_volume[i, time + 1] > tumour_death_threshold:
#                 cancer_volume[i, time + 1] = tumour_death_threshold
#                 sequence_lengths[i] = int(time+1)
#                 break  # patient death
#             elif cancer_volume[i, time + 1] <= 0:
#                 cancer_volume[i, time + 1] = 0
#                 sequence_lengths[i] = int(time+1)
#                 #break  # patient death
#             else:
#                 sequence_lengths[i] = max_time
                
                
            
#     simulated_real_data = {
#         "cancer_volume": cancer_volume,
#         "chemo_dosage": chemo_dosage,
#         "radio_dosage": radio_dosage,
#         "chemo_application": chemo_application_point,
#         "radio_application": radio_application_point,
#         "sequence_lengths": sequence_lengths,
#         "toxicity": toxic,
#         "patient_types": patient_types,
#         "simulation_params": simulation_params
#     }
    
#     processed_simulated_real_data = process_data(simulated_real_data.copy(),toxicity=True,continuous=True,scaling_data=scaling_data,treatment_testdata=True)
    
#     return simulated_real_data, processed_simulated_real_data
    
#     #volume auf 50 setzen als Minimum für unser Bsp. führt zu Durchmesser von 4.5, damit Dynamik vorhanden ist. 
#     #Vielleicht nochmal training?!

#     # scaling_data=pickle_map["scaling_data"]
#     # simulated_real_data, processed_simulated_real_data=simulate_true_model(0, output_earlier=None, initial=True, scaling_data=scaling_data, max_time=10, chemo=0,radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200)
#     # #outputs["current_treatments"][0,0,:]=[2,2]
#     # simulated_real_data, processed_simulated_real_data=simulate_true_model(0, output_earlier=simulated_real_data, initial=False, scaling_data=scaling_data, max_time=10, chemo=0,radio=0, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200)
#     # simulated_real_data, processed_simulated_real_data=simulate_true_model(1, output_earlier=simulated_real_data, initial=False, scaling_data=scaling_data, max_time=10, chemo=4,radio=1.5, num_patients=1, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200)
#     # #Unter 0 abfangen und mehrere Patienten einführen

# #fuer alle Behandlungsaenderungenoptionen
# #def simulate_test_allsol(num_time_steps, chemo_options=[0,3,5,7],radio_options=[0,1.5,1.75,2], num_patients=1, scaling_data=None, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200):
# def simulate_real_model_allsol(num_time_steps, chemo_options=[0,3,5,7],radio_options=[0,1.5,1.75,2], num_patients=1, scaling_data=None, chemo_coeff=4, radio_coeff=4, window_size=15, seed=200):
        
#     # Routine to search for opimal treatment options by testing all possible solutions
    
#     """
#     Core routine to generate simulation paths

#     :param simulation_params:
#     :param num_time_steps:
#     :param assigned_actions:
#     :return:
#     """

#     np.random.seed(seed)
    
#     simulation_params = get_confounding_params(
#         int(num_patients),
#         chemo_coeff=chemo_coeff,
#         radio_coeff=radio_coeff,
#         toxicity=True
#     )
#     simulation_params["window_size"] = window_size

#     total_num_chemo_treatments = 1

#     chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

#     # sort this
#     chemo_idx = np.argsort(chemo_days)
#     chemo_days = np.array(chemo_days)[chemo_idx]
    
#     chemo_amt=chemo_options
#     radio_amt=radio_options
#     treatment_options=list(itertools.product(chemo_amt,radio_amt))

#     drug_half_life = 1  # one day half life for drugs

#     # Unpack simulation parameters
#     initial_stages = simulation_params["initial_stages"]
#     initial_volumes = simulation_params["initial_volumes"]
#     alphas = simulation_params["alpha"]
#     rhos = simulation_params["rho"]
#     betas = simulation_params["beta"]
#     beta_cs = simulation_params["beta_c"]
#     Ks = simulation_params["K"]

#     num_patients = initial_stages.shape[0]
    
#     #extrem viele Optionen, vermutlich müssen wir das kürzen
#     #das geht nichtmal mit 10 Zeitschritten.... entweder weniger
#     #Zeitschritte oder weniger Behandlungen. Mit 10 Patienten geht das
#     number_options = (len(chemo_amt)*len(radio_amt))**num_time_steps
    
#     #patients*options*time_steps
#     cancer_volume = np.zeros((num_patients, number_options, num_time_steps+1))
#     cancer_volume[:]=np.nan
#     chemo_dosage = np.zeros((num_patients, number_options, num_time_steps))
#     chemo_dosage[:]=np.nan
#     radio_dosage = np.zeros((num_patients, number_options, num_time_steps))
#     radio_dosage[:]=np.nan
#     chemo_application_point = np.zeros((num_patients, number_options, num_time_steps))
#     chemo_application_point[:]=np.nan
#     radio_application_point = np.zeros((num_patients, number_options, num_time_steps))
#     radio_application_point[:]=np.nan
#     sequence_lengths = np.zeros((num_patients, number_options))
#     sequence_lengths[:]=np.nan
    
#     initial_toxicity = simulation_params["initial_toxicity"]
#     kgs = simulation_params["kg"]
#     kl1s = simulation_params["kl1"]
#     kl2s = simulation_params["kl2"]
#     kl3s = simulation_params["kl3"]
#     toxic = np.zeros((num_patients, number_options, num_time_steps+1))
#     toxic[:]=np.nan
#     noise_terms_toxic = 0.0015 * np.random.randn(
#         num_patients,
#         num_time_steps,
#     )
#     patient_types = simulation_params["patient_types"]

#     noise_terms = 0.01 * np.random.randn(
#         num_patients,
#         num_time_steps,
#     )  # 5% cell variability
    
#     recovery_rvs = np.random.rand(num_patients, num_time_steps+1)
    
#     for i in range(num_patients):
#         #if i % 200 == 0:
#         noise = noise_terms[i]        

#         # initial values
#         cancer_volume[i, :, 0] = initial_volumes[i]
#         alpha = alphas[i]
#         beta = betas[i]
#         beta_c = beta_cs[i]
#         rho = rhos[i]
#         K = Ks[i]
#         noise_toxic = noise_terms_toxic[i]
#         toxic[i,:, 0] = initial_toxicity[i]
#         kg=kgs[i]
#         kl1=kl1s[i]
#         kl2=kl2s[i]
#         kl3=kl3s[i]
        
#         for t in range(0, num_time_steps):
#             #outerloop for number of indices
#             index=int(number_options/((len(radio_amt)*len(chemo_amt))**(t+1)))
#             for j in range((len(radio_amt)*len(chemo_amt))**(t+1)):
#                 treatment_index=j % ((len(radio_amt)*len(chemo_amt)))
#                 #print(treatment_index)
#                 #chem = treatment_options[j][0]
#                 #radio = treatment_options[j][1]
#                 current_chemo_dose = 0.0
#                 previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, j*index, t - 1]
                
#                 # Action application
#                 radio_application_point[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][1]
#                 radio_dosage[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][1]

#                 current_chemo_dose = treatment_options[treatment_index][0]
#                 chemo_application_point[i, j*index:(j+1)*index, t] = treatment_options[treatment_index][0]
#                 # Update chemo dosage
#                 chemo_dosage[i, j*index:(j+1)*index, t] = (
#                     previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
#                     + current_chemo_dose
#                 )
    
#                 cancer_volume[i, j*index:(j+1)*index, t + 1] = cancer_volume[i, j*index:(j+1)*index, t] * (
#                     1
#                     + rho * np.log(K / cancer_volume[i, j*index:(j+1)*index, t])
#                     - beta_c * chemo_dosage[i, j*index:(j+1)*index, t]
#                     - (alpha * radio_dosage[i, j*index:(j+1)*index, t] + beta * radio_dosage[i, j*index:(j+1)*index, t] ** 2)
#                     + noise[t]
#                 )  # add noise to fit residuals
                    
#                 toxic[i, j*index:(j+1)*index, t + 1] = toxic[i, j*index:(j+1)*index, t] * (
#                     1
#                     + kg * toxic[i, j*index:(j+1)*index, t] *(1-toxic[i, j*index:(j+1)*index,t]/(initial_toxicity[i])) 
#                     - kl1 * chemo_dosage[i, j*index:(j+1)*index, t]
#                     - kl2 * radio_dosage[i, j*index:(j+1)*index, t]
#                     - kl3 * cancer_volume[i, j*index:(j+1)*index, t]
#                     + noise_toxic[t]
#                 )
#                 tumour_death_threshold=calc_volume(13)
#                 if cancer_volume[i, j*index, t + 1] > tumour_death_threshold:
#                     #print(i)
                    
#                     cancer_volume[i, j*index:(j+1)*index, t + 1] = tumour_death_threshold
#                     break  # patient death
                    
#                 if cancer_volume[i, j*index, t + 1] <= 0:
#                     #print(i)
                    
#                     cancer_volume[i, j*index:(j+1)*index, t + 1] = 0
#                     #break  # patient death
                                

#                 #if recovery_rvs[i, t + 1]/50 > cancer_volume[i, j*index, t + 1]: #np.exp(
#                 #    #-cancer_volume[i, t + 1] * tumour_cell_density,
#                 #):
#                 #    cancer_volume[i, j*index:(j+1)*index, t + 1] = 0
#                 #    break
                
#                 sequence_lengths[i, j*index:(j+1)*index] = int(t + 1)

#     #cancer_volume=np.delete(cancer_volume,pd.isna(cancer_volume[:,0]),axis=0)
#     #chemo_dosage=np.delete(chemo_dosage,pd.isna(cancer_volume[:,0]),axis=0)
#     #radio_dosage=np.delete(radio_dosage,pd.isna(cancer_volume[:,0]),axis=0)
#     #chemo_application_point=np.delete(chemo_application_point,pd.isna(cancer_volume[:,0]),axis=0)
#     #radio_application_point=np.delete(radio_application_point,pd.isna(cancer_volume[:,0]),axis=0)
#     #sequence_lengths=np.delete(sequence_lengths,pd.isna(sequence_lengths),axis=0)
    
#     simulated_real_data = {
#         "cancer_volume": cancer_volume,
#         "chemo_dosage": chemo_dosage,
#         "radio_dosage": radio_dosage,
#         "chemo_application": chemo_application_point,
#         "radio_application": radio_application_point,
#         "sequence_lengths": sequence_lengths,
#         "toxicity": toxic,
#         "patient_types": patient_types
#     }
#     #processed_simulated_real_data = process_data(simulated_real_data.copy(),toxicity=True,continuous=True,scaling_data=scaling_data,treatment_testdata=True)
#     return simulated_real_data



#### new from Causal Transformer ####
def simulate_counterfactual_1_step(simulation_params, seq_length, toxicity=False,continuous_therapy = False):
    """
    Simulation of test trajectories to asses all one-step ahead counterfactuals
    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length (number of factual time-steps)
    :return: simulated data dict with number of rows equal to num_patients * seq_length * num_treatments
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    num_treatments = 4  # No treatment / Chemotherapy / Radiotherapy / Chemotherapy + Radiotherapy

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]
    
    
    if continuous_therapy:
        chemo_amt=[3,5,7]
        radio_amt=[1,2,3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy
    #lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    num_test_points = num_patients * seq_length * num_treatments

    # Commence Simulation
    cancer_volume = np.zeros((num_test_points, seq_length))
    chemo_application_point = np.zeros((num_test_points, seq_length))
    radio_application_point = np.zeros((num_test_points, seq_length))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all_trajectories = np.zeros(num_test_points)
    
    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, seq_length))
        noise_terms_toxic = 0.0015 * np.random.randn(num_patients,seq_length,)    
    

    test_idx = 0

    # Run actual simulation
    for i in tqdm(range(num_patients), total=num_patients):

        # if i % 200 == 0:
        #     logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = 0.01 * np.random.randn(seq_length)  # 5% cell variability
        recovery_rvs = np.random.rand(seq_length)

        #initial values
        factual_cancer_volume = np.zeros(seq_length)
        factual_chemo_dosage = np.zeros(seq_length)
        factual_radio_dosage = np.zeros(seq_length)
        factual_chemo_application_point = np.zeros(seq_length)
        factual_radio_application_point = np.zeros(seq_length)
        factual_chemo_probabilities = np.zeros(seq_length)
        factual_radio_probabilities = np.zeros(seq_length)
        


        chemo_application_rvs = np.random.rand(seq_length)
        radio_application_rvs = np.random.rand(seq_length)
        
        
        if continuous_therapy:
            chemo_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])
            radio_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])

        factual_cancer_volume[0] = initial_volumes[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        
        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]
        
            #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7


        for t in range(0, seq_length - 1):

            # Factual prev_treatments and outcomes
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            # Action probabilities + death or recovery simulations
            # if t >= lag:
            #     cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
            # else:
            cancer_volume_used = np.zeros((1, ))
            #cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            factual_chemo_probabilities[t] = chemo_prob
            factual_radio_probabilities[t] = radio_prob
            
            
                  # # # Action application
                  # if radio_application_rvs[t] < radio_prob:
                  #     factual_radio_application_point[t] = 1
                  #     factual_radio_dosage[t] = radio_amt[0]

                  # if chemo_application_rvs[t] < chemo_prob:
                  #     factual_chemo_application_point[t] = 1
                  #     current_chemo_dose = chemo_amt[0]            


            #Action application
            if radio_application_rvs[t] < radio_prob:
                if continuous_therapy:
                    if radio_prob+radio_dosage_rvs[i,t]<0.25:
                        factual_radio_dosage[t] = radio_amt[0]
                    elif radio_prob+radio_dosage_rvs[i,t]>0.25 and radio_prob+radio_dosage_rvs[i,t]<0.5:
                        factual_radio_dosage[t] = radio_amt[1]
                    else:
                        factual_radio_dosage[t] = radio_amt[2]
                    factual_radio_application_point[t] = factual_radio_dosage[t]
                        
                else:
                    factual_radio_application_point[t] = 1
                    factual_radio_dosage[t] = radio_amt[0]

            if chemo_application_rvs[t] < chemo_prob:
                
                #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
                if continuous_therapy:
                    if chemo_prob+chemo_dosage_rvs[i,t]<0.25:
                        current_chemo_dose = chemo_amt[0]
                    elif chemo_prob+chemo_dosage_rvs[i,t]>0.25 and chemo_prob+chemo_dosage_rvs[t]<0.5:
                        current_chemo_dose = chemo_amt[1]
                    else:
                        current_chemo_dose = chemo_amt[2]
                    factual_chemo_application_point[t] = current_chemo_dose
                        
                else:
                    #wie transformer 
                    factual_chemo_application_point[t] = 1
                    current_chemo_dose = chemo_amt[0]                   
                    
                    
                  ####
                  

            # Update chemo dosage
            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # Factual prev_treatments and outcomes
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
                (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * factual_chemo_dosage[t] -
                    (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])

            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

            # Populate arrays
            cancer_volume[test_idx] = factual_cancer_volume
            chemo_application_point[test_idx] = factual_chemo_application_point
            radio_application_point[test_idx] = factual_radio_application_point
            patient_types_all_trajectories[test_idx] = patient_types[i]
            sequence_lengths[test_idx] = int(t) + 1
            test_idx = test_idx + 1

            # Counterfactual prev_treatments and outcomes
            treatment_options = [(0, 0), (0, 1), (1, 0), (1, 1)]  # First = chemo; second = radio

            for treatment_option in treatment_options:
                if (factual_chemo_application_point[t] == treatment_option[0] and factual_radio_application_point[t] ==
                        treatment_option[1]):
                    # This represents the factual treatment which was already considered
                    continue
                current_chemo_dose = 0.0
                counterfactual_radio_dosage = 0.0
                counterfactual_chemo_application_point = 0
                counterfactual_radio_application_point = 0

                if treatment_option[0] == 1:
                    counterfactual_chemo_application_point = 1
                    current_chemo_dose = chemo_amt[0]

                if treatment_option[1] == 1:
                    counterfactual_radio_application_point = 1
                    counterfactual_radio_dosage = radio_amt[0]

                counterfactual_chemo_dosage = previous_chemo_dose * np.exp(
                    -np.log(2) / drug_half_life) + current_chemo_dose

                counterfactual_cancer_volume = factual_cancer_volume[t] *\
                    (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * counterfactual_chemo_dosage -
                        (alpha * counterfactual_radio_dosage + beta * counterfactual_radio_dosage ** 2) + noise[t + 1])

                cancer_volume[test_idx][:t + 2] = np.append(factual_cancer_volume[:t + 1],
                                                            [counterfactual_cancer_volume])
                chemo_application_point[test_idx][:t + 1] = np.append(factual_chemo_application_point[:t],
                                                                      [counterfactual_chemo_application_point])
                radio_application_point[test_idx][:t + 1] = np.append(factual_radio_application_point[:t],
                                                                      [counterfactual_radio_application_point])
                patient_types_all_trajectories[test_idx] = patient_types[i]
                sequence_lengths[test_idx] = int(t) + 1
                test_idx = test_idx + 1
                
                
                if toxicity:
                    toxic[ t + 1] = toxic[t] * (1
                        + kg * toxic[t] *(1-toxic[t]) 
                        - kl1 * counterfactual_chemo_dosage[t]
                        - kl2 * counterfactual_radio_dosage[t]
                        - kl3 * cancer_volume[t]
                        + noise_toxic[t]
                    )

            if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
                    recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
                break
    # if toxicity:
        
    #     outputs = {'cancer_volume': cancer_volume[:test_idx],
    #                'chemo_application': chemo_application_point[:test_idx],
    #                'radio_application': radio_application_point[:test_idx],
    #                'sequence_lengths': sequence_lengths[:test_idx],
    #                'patient_types': patient_types_all_trajectories[:test_idx],
    #                "toxicity": toxic
    #                }
    # else:
    #      outputs = {'cancer_volume': cancer_volume[:test_idx],
    #                 'chemo_application': chemo_application_point[:test_idx],
    #                 'radio_application': radio_application_point[:test_idx],
    #                 'sequence_lengths': sequence_lengths[:test_idx],
    #                 'patient_types': patient_types_all_trajectories[:test_idx]
                    
    #                 }
         
    if toxicity:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
            "toxicity": toxic
        }
        
    else:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
        }         

    print("Call to simulate counterfactuals data")

    return outputs


def simulate_counterfactuals_treatment_seq(simulation_params, seq_length, projection_horizon, cf_seq_mode='sliding_treatment',toxicity=False,continuous_therapy=False):
    """
    Simulation of test trajectories to asses a subset of multiple-step ahead counterfactuals
    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length (number of factual time-steps)
    :param cf_seq_mode: Counterfactual sequence setting: sliding_treatment / random_trajectories
    :return: simulated data dict with number of rows equal to num_patients * seq_length * 2 * projection_horizon
    """

    if cf_seq_mode == 'sliding_treatment':
        chemo_arr = np.stack([np.eye(projection_horizon, dtype=int),
                              np.zeros((projection_horizon, projection_horizon), dtype=int)], axis=-1)
        radio_arr = np.stack([np.zeros((projection_horizon, projection_horizon), dtype=int),
                              np.eye(projection_horizon, dtype=int)], axis=-1)
        treatment_options = np.concatenate([chemo_arr, radio_arr])
    elif cf_seq_mode == 'random_trajectories':
        treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 2))
    else:
        raise NotImplementedError()

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]
    
    if continuous_therapy:
        chemo_amt=[3,5,7]
        radio_amt=[1,2,3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy
    #lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    num_test_points = len(treatment_options) * num_patients * seq_length

    # Commence Simulation
    cancer_volume = np.zeros((num_test_points, seq_length + projection_horizon))
    chemo_application_point = np.zeros((num_test_points, seq_length + projection_horizon))
    radio_application_point = np.zeros((num_test_points, seq_length + projection_horizon))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all_trajectories = np.zeros(num_test_points)
    patient_ids_all_trajectories = np.zeros(num_test_points)
    patient_current_t = np.zeros(num_test_points)
    
    
    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, seq_length))
        noise_terms_toxic = 0.0015 * np.random.randn(
                num_patients,
                seq_length,
            )    

    test_idx = 0

    # Run actual simulation
    for i in tqdm(range(num_patients), total=num_patients):

        # if i % 200 == 0:
        #     logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = 0.01 * np.random.randn(seq_length + projection_horizon)  # 5% cell variability
        recovery_rvs = np.random.rand(seq_length)

        # initial values
        factual_cancer_volume = np.zeros(seq_length)
        factual_chemo_dosage = np.zeros(seq_length)
        factual_radio_dosage = np.zeros(seq_length)
        factual_chemo_application_point = np.zeros(seq_length)
        factual_radio_application_point = np.zeros(seq_length)
        factual_chemo_probabilities = np.zeros(seq_length)
        factual_radio_probabilities = np.zeros(seq_length)

        chemo_application_rvs = np.random.rand(seq_length)
        radio_application_rvs = np.random.rand(seq_length)
        
        
        if continuous_therapy:
            chemo_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])
            radio_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])

        factual_cancer_volume[0] = initial_volumes[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        
        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]

        for t in range(0, seq_length - 1):

            # Factual prev_treatments and outcomes
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            # Action probabilities + death or recovery simulations
            # if t >= lag:
            #     cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
            # else:
            cancer_volume_used = np.zeros((1,))
            #cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            factual_chemo_probabilities[t] = chemo_prob
            factual_radio_probabilities[t] = radio_prob

            
            if radio_application_rvs[t] < radio_prob:
                if continuous_therapy:
                    if radio_prob+radio_dosage_rvs[i,t]<0.25:
                        factual_radio_dosage[t] = radio_amt[0]
                    elif radio_prob+radio_dosage_rvs[i,t]>0.25 and radio_prob+radio_dosage_rvs[i,t]<0.5:
                        factual_radio_dosage[t] = radio_amt[1]
                    else:
                        factual_radio_dosage[t] = radio_amt[2]
                    factual_radio_application_point[t] = factual_radio_dosage[t]
                        
                else:
                    factual_radio_application_point[t] = 1
                    factual_radio_dosage[t] = radio_amt[0]

            if chemo_application_rvs[t] < chemo_prob:
                
                #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
                if continuous_therapy:
                    if chemo_prob+chemo_dosage_rvs[i,t]<0.25:
                        current_chemo_dose = chemo_amt[0]
                    elif chemo_prob+chemo_dosage_rvs[i,t]>0.25 and chemo_prob+chemo_dosage_rvs[t]<0.5:
                        current_chemo_dose = chemo_amt[1]
                    else:
                        current_chemo_dose = chemo_amt[2]
                    factual_chemo_application_point[t] = current_chemo_dose
                        
                else:
                    #wie transformer 
                    factual_chemo_application_point[t] = 1
                    current_chemo_dose = chemo_amt[0]               
            
            
                       

            # Update chemo dosage
            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # Factual prev_treatments and outcomes
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
                (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * factual_chemo_dosage[t] -
                    (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])

            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

            if cf_seq_mode == 'random_trajectories':
                treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 2))

            for treatment_option in treatment_options:

                counterfactual_cancer_volume = np.zeros(shape=(t + 1 + projection_horizon + 1))
                counterfactual_chemo_application_point = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_radio_application_point = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_chemo_dosage = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_radio_dosage = np.zeros(shape=(t + 1 + projection_horizon))

                counterfactual_cancer_volume[:t + 2] = factual_cancer_volume[:t + 2]
                counterfactual_chemo_application_point[:t + 1] = factual_chemo_application_point[:t + 1]
                counterfactual_radio_application_point[:t + 1] = factual_radio_application_point[:t + 1]
                counterfactual_chemo_dosage[:t + 1] = factual_chemo_dosage[:t + 1]
                counterfactual_radio_dosage[:t + 1] = factual_radio_dosage[:t + 1]

                for projection_time in range(0, projection_horizon):

                    current_t = t + 1 + projection_time
                    previous_chemo_dose = counterfactual_chemo_dosage[current_t - 1]

                    current_chemo_dose = 0.0
                    counterfactual_radio_dosage[current_t] = 0.0
                    if treatment_option[projection_time][0] == 1:
                        counterfactual_chemo_application_point[current_t] = 1
                        current_chemo_dose = chemo_amt[0]

                    if treatment_option[projection_time][1] == 1:
                        counterfactual_radio_application_point[current_t] = 1
                        counterfactual_radio_dosage[current_t] = radio_amt[0]

                    counterfactual_chemo_dosage[current_t] = previous_chemo_dose * np.exp(
                        -np.log(2) / drug_half_life) + current_chemo_dose

                    counterfactual_cancer_volume[current_t + 1] = counterfactual_cancer_volume[current_t] *\
                        (1 + rho * np.log(K / (counterfactual_cancer_volume[current_t] + 1e-07) + 1e-07) -
                         beta_c * counterfactual_chemo_dosage[current_t] -
                         (alpha * counterfactual_radio_dosage[current_t] + beta * counterfactual_radio_dosage[current_t] ** 2) +
                         noise[current_t + 1])

                if (np.isnan(counterfactual_cancer_volume).any()):
                    continue

                cancer_volume[test_idx][:t + 1 + projection_horizon + 1] = counterfactual_cancer_volume
                chemo_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_chemo_application_point
                radio_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_radio_application_point
                patient_types_all_trajectories[test_idx] = patient_types[i]
                patient_ids_all_trajectories[test_idx] = i
                patient_current_t[test_idx] = t

                sequence_lengths[test_idx] = int(t) + projection_horizon + 1
                test_idx = test_idx + 1
                
                
                if toxicity:
                    toxic[i, t + 1] = toxic[i, t] * (
                        1
                        + kg * toxic[i, t] *(1-toxic[i,t]/(initial_toxicity[i])) 
                        - kl1 * counterfactual_chemo_dosage[i, t]
                        - kl2 * counterfactual_radio_dosage[i, t]
                        - kl3 * cancer_volume[i, t]
                        + noise_toxic[t]
                    )    

            if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
                    recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
                break


    
    # if toxicity:
        
    #     outputs = {'cancer_volume': cancer_volume[:test_idx],
    #                'chemo_application': chemo_application_point[:test_idx],
    #                'radio_application': radio_application_point[:test_idx],
    #                'sequence_lengths': sequence_lengths[:test_idx],
    #                'patient_types': patient_types_all_trajectories[:test_idx],
    #                'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
    #                'patient_current_t': patient_current_t[:test_idx],
    #                "toxicity": toxic
    #                }
    # else:
    #     outputs = {'cancer_volume': cancer_volume[:test_idx],
    #                'chemo_application': chemo_application_point[:test_idx],
    #                'radio_application': radio_application_point[:test_idx],
    #                'sequence_lengths': sequence_lengths[:test_idx],
    #                'patient_types': patient_types_all_trajectories[:test_idx],
    #                'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
    #                'patient_current_t': patient_current_t[:test_idx],
    #                }
    
    
    if toxicity:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
            "toxicity": toxic
        }
        
    else:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
        }

    # print("Call to simulate counterfactuals data")

    return outputs

####


def simulate_counterfactual_1_step1(simulation_params, seq_length, toxicity=False,continuous_therapy = False):
    """
    Simulation of test trajectories to asses all one-step ahead counterfactuals
    :param simulation_params: Parameters of the simulation
    :param seq_length: Maximum trajectory length (number of factual time-steps)
    :return: simulated data dict with number of rows equal to num_patients * seq_length * num_treatments
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    num_treatments = 4  # No treatment / Chemotherapy / Radiotherapy / Chemotherapy + Radiotherapy

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]
    
    
    if continuous_therapy:
        chemo_amt=[3,5,7]
        radio_amt=[1,2,3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy
    #lag = simulation_params['lag']

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    num_test_points = num_patients * seq_length * num_treatments

    # Commence Simulation
    cancer_volume = np.zeros((num_test_points, seq_length))
    chemo_application_point = np.zeros((num_test_points, seq_length))
    radio_application_point = np.zeros((num_test_points, seq_length))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all_trajectories = np.zeros(num_test_points)
    
    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, seq_length))
        noise_terms_toxic = 0.0015 * np.random.randn(num_patients,seq_length,)    
    

    test_idx = 0

    # Run actual simulation
    for i in tqdm(range(num_patients), total=num_patients):

        # if i % 200 == 0:
        #     logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = 0.01 * np.random.randn(seq_length)  # 5% cell variability
        recovery_rvs = np.random.rand(seq_length)

        # initial values
        # factual_cancer_volume = np.zeros(seq_length)
        # factual_chemo_dosage = np.zeros(seq_length)
        # factual_radio_dosage = np.zeros(seq_length)
        # factual_chemo_application_point = np.zeros(seq_length)
        # factual_radio_application_point = np.zeros(seq_length)
        # factual_chemo_probabilities = np.zeros(seq_length)
        # factual_radio_probabilities = np.zeros(seq_length)
        
        # Commence Simulation
        factual_cancer_volume = np.zeros((num_patients,seq_length))
        factual_chemo_dosage = np.zeros((num_patients, seq_length))#sb
        factual_radio_dosage = np.zeros((num_patients, seq_length))#sb
        factual_chemo_application_point = np.zeros((num_patients,seq_length))
        factual_radio_application_point = np.zeros((num_patients, seq_length))
        factual_sequence_lengths = np.zeros(num_patients)
        factual_chemo_probabilities = np.zeros((num_patients, seq_length))#sb
        factual_radio_probabilities = np.zeros((num_patients, seq_length))#sb
        

        # chemo_application_rvs = np.random.rand(seq_length)
        # radio_application_rvs = np.random.rand(seq_length)
        
        chemo_application_rvs = np.random.rand(num_patients, seq_length)#sb
        radio_application_rvs = np.random.rand(num_patients, seq_length)#sb
        
        
        if continuous_therapy:
            chemo_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])
            radio_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])

        factual_cancer_volume[0] = initial_volumes[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        
        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg=kgs[i]
            kl1=kl1s[i]
            kl2=kl2s[i]
            kl3=kl3s[i]
        
            #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7


        for t in range(0, seq_length - 1):

            # Factual prev_treatments and outcomes
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            # Action probabilities + death or recovery simulations
            # if t >= lag:
            #     cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
            # else:
            cancer_volume_used = np.zeros((1, ))
            #cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            factual_chemo_probabilities[t] = chemo_prob
            factual_radio_probabilities[t] = radio_prob

            # # # Action application
            # if radio_application_rvs[t] < radio_prob:
            #     factual_radio_application_point[t] = 1
            #     factual_radio_dosage[t] = radio_amt[0]

            # if chemo_application_rvs[t] < chemo_prob:
            #     factual_chemo_application_point[t] = 1
            #     current_chemo_dose = chemo_amt[0]
                
            #Action application
            if radio_application_rvs[t] < radio_prob:
                factual_radio_application_point[t] = 1
                if continuous_therapy:
                    #print('con')
                    if radio_prob+radio_dosage_rvs[t]<0.25:
                        factual_radio_dosage[t] = radio_amt[0]
                    elif radio_prob+radio_dosage_rvs[t]>0.25 and radio_prob+radio_dosage_rvs[t]<0.5:
                        factual_radio_dosage[t] = radio_amt[1]
                    else:
                        factual_radio_dosage[t] = radio_amt[2]
                    radio_application_point[t] = factual_radio_dosage[i,t]
                        
                else:
                    #print('not con')
                    factual_radio_dosage[t] = radio_amt[0]

            if chemo_application_rvs[t] < chemo_prob:
                factual_chemo_application_point[t] = 1
                #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
                if continuous_therapy:
                    if chemo_prob+chemo_dosage_rvs[t]<0.25:
                        current_chemo_dose = chemo_amt[0]
                    elif chemo_prob+chemo_dosage_rvs[t]>0.25 and chemo_prob+chemo_dosage_rvs[t]<0.5:
                        current_chemo_dose = chemo_amt[1]
                    else:
                        current_chemo_dose = chemo_amt[2]
                    chemo_application_point[t] = current_chemo_dose
                        
                else:
                    current_chemo_dose = chemo_amt[0]                           
                
            
                
            # # Action application
            # if radio_application_rvs[i, t] < radio_prob:
            #     factual_radio_application_point[i,t] = 1
            #     if continuous_therapy:
            #         if radio_prob+radio_dosage_rvs[i,t]<0.25:
            #             factual_radio_dosage[i, t] = radio_amt[0]
            #         elif radio_prob+radio_dosage_rvs[i,t]>0.25 and radio_prob+radio_dosage_rvs[i,t]<0.5:
            #             factual_radio_dosage[i, t] = radio_amt[1]
            #         else:
            #             factual_radio_dosage[i, t] = radio_amt[2]
            #         radio_application_point[i, t] = factual_radio_dosage[i,t]
                        
            #     else:
            #         factual_radio_application_point[t] = 1
            #         factual_radio_dosage[i, t] = radio_amt[0]

            # if chemo_application_rvs[i, t] < chemo_prob:
            #     factual_chemo_application_point[i, t] = 1
            #     #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
            #     if continuous_therapy:
            #         if chemo_prob+chemo_dosage_rvs[i,t]<0.25:
            #             current_chemo_dose = chemo_amt[0]
            #         elif chemo_prob+chemo_dosage_rvs[i,t]>0.25 and chemo_prob+chemo_dosage_rvs[i,t]<0.5:
            #             current_chemo_dose = chemo_amt[1]
            #         else:
            #             current_chemo_dose = chemo_amt[2]
            #         chemo_application_point[i, t] = current_chemo_dose
                        
            #     else:
            #         factual_chemo_application_point[t] = 1
            #         current_chemo_dose = chemo_amt[0]    
                

            # Update chemo dosage
            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # Factual prev_treatments and outcomes
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
                (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * factual_chemo_dosage[t] -
                    (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])

            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

            # Populate arrays
            cancer_volume[test_idx] = factual_cancer_volume
            chemo_application_point[test_idx] = factual_chemo_application_point
            radio_application_point[test_idx] = factual_radio_application_point
            patient_types_all_trajectories[test_idx] = patient_types[i]
            sequence_lengths[test_idx] = int(t) + 1
            test_idx = test_idx + 1

            # Counterfactual prev_treatments and outcomes
            treatment_options = [(0, 0), (0, 1), (1, 0), (1, 1)]  # First = chemo; second = radio

            for treatment_option in treatment_options:
                if (factual_chemo_application_point[t] == treatment_option[0] and factual_radio_application_point[t] ==
                        treatment_option[1]):
                    # This represents the factual treatment which was already considered
                    continue
                current_chemo_dose = 0.0
                counterfactual_radio_dosage = 0.0
                counterfactual_chemo_application_point = 0
                counterfactual_radio_application_point = 0

                if treatment_option[0] == 1:
                    counterfactual_chemo_application_point = 1
                    current_chemo_dose = chemo_amt[0]

                if treatment_option[1] == 1:
                    counterfactual_radio_application_point = 1
                    counterfactual_radio_dosage = radio_amt[0]

                counterfactual_chemo_dosage = previous_chemo_dose * np.exp(
                    -np.log(2) / drug_half_life) + current_chemo_dose

                counterfactual_cancer_volume = factual_cancer_volume[t] *\
                    (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * counterfactual_chemo_dosage -
                        (alpha * counterfactual_radio_dosage + beta * counterfactual_radio_dosage ** 2) + noise[t + 1])

                cancer_volume[test_idx][:t + 2] = np.append(factual_cancer_volume[:t + 1],
                                                            [counterfactual_cancer_volume])
                chemo_application_point[test_idx][:t + 1] = np.append(factual_chemo_application_point[:t],
                                                                      [counterfactual_chemo_application_point])
                radio_application_point[test_idx][:t + 1] = np.append(factual_radio_application_point[:t],
                                                                      [counterfactual_radio_application_point])
                patient_types_all_trajectories[test_idx] = patient_types[i]
                sequence_lengths[test_idx] = int(t) + 1
                test_idx = test_idx + 1
                
                
                # if toxicity:
                #     toxic[ t + 1] = toxic[t] * (1
                #         + kg * toxic[t] *(1-toxic[t]) 
                #         - kl1 * counterfactual_chemo_dosage[t]
                #         - kl2 * counterfactual_radio_dosage[t]
                #         - kl3 * cancer_volume[t]
                #         + noise_toxic[t]
                #     )
                if toxicity: 
                    toxic[i, t + 1] = toxic[i, t] * (1
                        + kg * toxic[i, t] *(1- toxic[i,t]/(initial_toxicity[i])) 
                        - kl1 * counterfactual_chemo_dosage[i,t]
                        - kl2 * counterfactual_radio_dosage[i,t]
                        - kl3 * cancer_volume[i,t]
                        + noise_toxic[t])
                
                # toxic[i, t + 1] = toxic[i, t] * (1
                #     + kg * toxic[i, t] *(1- toxic[i,t]/(initial_toxicity[i])) 
                #     - kl1 * counterfactual_chemo_dosage[i, t]
                #     - kl2 * counterfactual_radio_dosage[i, t]
                #     - kl3 * cancer_volume[i, t]
                #     + noise_toxic[t])

            if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
                    recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
                break
    if toxicity:
        
        outputs = {'cancer_volume': cancer_volume[:test_idx],
                   'chemo_application': chemo_application_point[:test_idx],
                   'radio_application': radio_application_point[:test_idx],
                   'sequence_lengths': sequence_lengths[:test_idx],
                   'patient_types': patient_types_all_trajectories[:test_idx],
                   "toxicity": toxic
                   }
    else:
         outputs = {'cancer_volume': cancer_volume[:test_idx],
                    'chemo_application': chemo_application_point[:test_idx],
                    'radio_application': radio_application_point[:test_idx],
                    'sequence_lengths': sequence_lengths[:test_idx],
                    'patient_types': patient_types_all_trajectories[:test_idx]
                    
                    }

    print("Call to simulate counterfactuals data")
    
    
    if toxicity:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
            "toxicity": toxic
        }
        
    else:
        outputs = {
            "cancer_volume": cancer_volume,
            "chemo_dosage": factual_chemo_dosage,
            "radio_dosage": factual_radio_dosage,
            "chemo_application": chemo_application_point,
            "radio_application": radio_application_point,
            "chemo_probabilities": factual_chemo_probabilities,
            "radio_probabilities": factual_radio_probabilities,
            "sequence_lengths": sequence_lengths,
            "patient_types": patient_types,
        }

    return outputs



def simulate_with_counterfactuals_my1(simulation_params, num_time_steps, assigned_actions=None,
                                  toxicity=False, continuous_therapy=False):
    """
    Simulation mit faktischen und kontrafaktischen Behandlungen (einschließlich Toxizität & kontinuierlichen Dosen)
    """

    # --- Behandlungsoptionen vorbereiten ---
    base_radio_amt = np.array([2.0])
    base_chemo_amt = np.array([5.0])
    if continuous_therapy:
        chemo_amt = [3, 5, 7]
        radio_amt = [1, 2, 3]
    else:
        chemo_amt = base_chemo_amt.tolist()
        radio_amt = base_radio_amt.tolist()
        
    # if continuous_therapy:
    #     treatment_options_chemo = [0] + chemo_amt  # z.B. [0, 3, 5, 7]
    #     treatment_options_radio = [0] + radio_amt  # z.B. [0, 1, 2, 3]
    # else:
    #     treatment_options_chemo = [0, chemo_amt[0]]  # [0, 5]
    #     treatment_options_radio = [0, radio_amt[0]]  # [0, 2]
    

    treatment_options_chemo = [0] + chemo_amt
    treatment_options_radio = [0] + radio_amt
    num_cf_chemo = len(treatment_options_chemo)
    num_cf_radio = len(treatment_options_radio)
    num_cf = num_cf_chemo * num_cf_radio  # Anzahl kontrafaktischer Szenarien (inkl. faktischem)

    drug_half_life = 1

    # --- Parameter auspacken ---
    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]

    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]

    num_patients = initial_volumes.shape[0]

    # --- Arrays für Faktisch + Kontrafaktisch ---
    cancer_volume_cf = np.zeros((num_patients, num_time_steps, num_cf))
    chemo_dosage_cf = np.zeros((num_patients, num_time_steps, num_cf))
    radio_dosage_cf = np.zeros((num_patients, num_time_steps, num_cf))
    chemo_application_cf = np.zeros((num_patients, num_time_steps, num_cf))
    radio_application_cf = np.zeros((num_patients, num_time_steps, num_cf))
    sequence_lengths_cf = np.zeros((num_patients, num_cf))
    if toxicity:
        toxic_cf = np.zeros((num_patients, num_time_steps, num_cf))

    noise_terms = 0.01 * np.random.randn(num_patients, num_time_steps)
    if toxicity:
        noise_terms_toxic = 0.0015 * np.random.randn(num_patients, num_time_steps)

    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    # --- Simulation starten ---
    for i in range(num_patients):
        noise = noise_terms[i]
        if toxicity:
            noise_toxic = noise_terms_toxic[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        if toxicity:
            kg = kgs[i]
            kl1 = kl1s[i]
            kl2 = kl2s[i]
            kl3 = kl3s[i]

        # Initialwerte setzen (alle Szenarien starten gleich)
        cancer_volume_cf[i, 0, :] = initial_volumes[i]
        if toxicity:
            toxic_cf[i, 0, :] = initial_toxicity[i]

        for t in range(num_time_steps - 1):
            # --- Jedes Szenario separat simulieren ---
            cf_idx = 0
            for chemo_choice in treatment_options_chemo:
                for radio_choice in treatment_options_radio:

                    # Faktisches Szenario (falls assigned_actions gegeben)
                    if assigned_actions is not None and cf_idx == 0:
                        current_chemo_dose = assigned_actions[i, t, 0]
                        current_radio_dose = assigned_actions[i, t, 1]
                    else:
                        if continuous_therapy:
                            # Softmax Wahrscheinlichkeiten -> Sampling
                            chemo_probs = softmax([j * chemo_sigmoid_betas[i] *
                                                   (calc_diameter(cancer_volume_cf[i, t, cf_idx]) -
                                                    chemo_sigmoid_intercepts[i])
                                                   for j in range(len(treatment_options_chemo))])
                            radio_probs = softmax([j * radio_sigmoid_betas[i] *
                                                   (calc_diameter(cancer_volume_cf[i, t, cf_idx]) -
                                                    radio_sigmoid_intercepts[i])
                                                   for j in range(len(treatment_options_radio))])
                            current_chemo_dose = chemo_choice
                            current_radio_dose = radio_choice
                        else:
                            current_chemo_dose = chemo_choice
                            current_radio_dose = radio_choice

                    # --- Dosierung updaten ---
                    prev_chemo_dose = chemo_dosage_cf[i, t-1, cf_idx] if t > 0 else 0
                    chemo_dosage_cf[i, t, cf_idx] = prev_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
                    radio_dosage_cf[i, t, cf_idx] = current_radio_dose
                    chemo_application_cf[i, t, cf_idx] = current_chemo_dose
                    radio_application_cf[i, t, cf_idx] = current_radio_dose

                    # --- Tumorvolumen ---
                    cancer_volume_cf[i, t+1, cf_idx] = cancer_volume_cf[i, t, cf_idx] * (
                        1 + rho * np.log(K / cancer_volume_cf[i, t, cf_idx])
                        - beta_c * chemo_dosage_cf[i, t, cf_idx]
                        - (alpha * radio_dosage_cf[i, t, cf_idx] + beta * radio_dosage_cf[i, t, cf_idx]**2)
                        + noise[t]
                    )

                    # --- Toxizität ---
                    if toxicity:
                        toxic_cf[i, t+1, cf_idx] = toxic_cf[i, t, cf_idx] * (
                            1 + kg * toxic_cf[i, t, cf_idx] * (1 - toxic_cf[i, t, cf_idx] / initial_toxicity[i])
                            - kl1 * chemo_dosage_cf[i, t, cf_idx]
                            - kl2 * radio_dosage_cf[i, t, cf_idx]
                            - kl3 * cancer_volume_cf[i, t, cf_idx]
                            + noise_toxic[t]
                        )

                    # --- Abbruchbedingungen ---
                    if cancer_volume_cf[i, t+1, cf_idx] > tumour_death_threshold:
                        cancer_volume_cf[i, t+1, cf_idx] = tumour_death_threshold
                        sequence_lengths_cf[i, cf_idx] = t+1
                        continue
                    if recovery_rvs[i, t+1] / 50 > cancer_volume_cf[i, t+1, cf_idx]:
                        cancer_volume_cf[i, t+1, cf_idx] = 0
                        sequence_lengths_cf[i, cf_idx] = t+1
                        continue

                    sequence_lengths_cf[i, cf_idx] = t+1
                    cf_idx += 1

    outputs = {
        "cancer_volume": cancer_volume_cf,
        "chemo_dosage": chemo_dosage_cf,
        "radio_dosage": radio_dosage_cf,
        "chemo_application": chemo_application_cf,
        "radio_application": radio_application_cf,
        "sequence_lengths": sequence_lengths_cf,
        "patient_types": patient_types
    }
    if toxicity:
        outputs["toxicity_cf"] = toxic_cf

    return outputs



def simulate_with_counterfactuals_myseq(
    simulation_params,
    num_time_steps,
    assigned_actions=None,
    toxicity=False,
    continuous_therapy=False,
    counterfactual_treatment_sequences=None,  # Neu
    projection_horizon=0,                      # Neu
    counterfactual_start_times=None            # Neu
):
    """
    Core routine to generate simulation paths with optional counterfactual simulations.
    """

    # [Dein bisheriger Code ...]
    # --------------------------
    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    if continuous_therapy:
        chemo_amt = [3, 5, 7]
        radio_amt = [1, 2, 3]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params["initial_stages"]
    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]

    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]

    num_patients = initial_stages.shape[0]

    # Arrays für Simulationsergebnisse
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    if toxicity:
        initial_toxicity = simulation_params["initial_toxicity"]
        kgs = simulation_params["kg"]
        kl1s = simulation_params["kl1"]
        kl2s = simulation_params["kl2"]
        kl3s = simulation_params["kl3"]
        toxic = np.zeros((num_patients, num_time_steps))
        noise_terms_toxic = 0.0015 * np.random.randn(num_patients, num_time_steps)

    noise_terms = 0.01 * np.random.randn(num_patients, num_time_steps)
    recovery_rvs = np.random.rand(num_patients, num_time_steps)
    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    if continuous_therapy:
        chemo_dosage_rvs = np.random.uniform(low=-0.05, high=0.05, size=[num_patients, num_time_steps])
        radio_dosage_rvs = np.random.uniform(low=-0.05, high=0.05, size=[num_patients, num_time_steps])

    # ---------------------------
    # Faktische Simulation (wie bisher)
    # ---------------------------
    for i in range(num_patients):
        if i % 200 == 0:
            logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = noise_terms[i]
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        if toxicity:
            noise_toxic = noise_terms_toxic[i]
            toxic[i, 0] = initial_toxicity[i]
            kg = kgs[i]
            kl1 = kl1s[i]
            kl2 = kl2s[i]
            kl3 = kl3s[i]

        for t in range(0, num_time_steps - 1):

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]

            cancer_volume_used = cancer_volume[i, max(t - window_size, 0): t + 1]
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used]).mean()
            cancer_metric_used = cancer_diameter_used

            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:
                radio_prob = 1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i])))
                chemo_prob = 1.0 / (1.0 + np.exp(-chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i])))

            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob:
                radio_application_point[i, t] = 1
                if continuous_therapy:
                    val = radio_prob + radio_dosage_rvs[i, t]
                    if val < 0.25:
                        radio_dosage[i, t] = radio_amt[0]
                    elif val < 0.5:
                        radio_dosage[i, t] = radio_amt[1]
                    else:
                        radio_dosage[i, t] = radio_amt[2]
                    radio_application_point[i, t] = radio_dosage[i, t]
                else:
                    radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:
                chemo_application_point[i, t] = 1
                if continuous_therapy:
                    val = chemo_prob + chemo_dosage_rvs[i, t]
                    if val < 0.25:
                        current_chemo_dose = chemo_amt[0]
                    elif val < 0.5:
                        current_chemo_dose = chemo_amt[1]
                    else:
                        current_chemo_dose = chemo_amt[2]
                    chemo_application_point[i, t] = current_chemo_dose
                else:
                    current_chemo_dose = chemo_amt[0]

            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            cancer_volume[i, t + 1] = cancer_volume[i, t] * (
                1
                + rho * np.log(K / cancer_volume[i, t])
                - beta_c * chemo_dosage[i, t]
                - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
                + noise[t]
            )

            if cancer_volume[i, t + 1] > tumour_death_threshold:
                cancer_volume[i, t + 1] = tumour_death_threshold
                break

            if recovery_rvs[i, t + 1] / 50 > cancer_volume[i, t + 1]:
                cancer_volume[i, t + 1] = 0
                break

            if toxicity:
                toxic[i, t + 1] = toxic[i, t] * (
                    1
                    + kg * toxic[i, t] * (1 - toxic[i, t] / initial_toxicity[i])
                    - kl1 * chemo_dosage[i, t]
                    - kl2 * radio_dosage[i, t]
                    - kl3 * cancer_volume[i, t]
                    + noise_toxic[t]
                )

        sequence_lengths[i] = int(t + 1)

    # ---------------------------
    # Counterfaktische Simulation
    # ---------------------------
    if (
        counterfactual_treatment_sequences is not None
        and projection_horizon > 0
        and counterfactual_start_times is not None
    ):
        # Dimensions: patient x treatment_sequence x projection_horizon+1
        num_cf_sequences = len(counterfactual_treatment_sequences)
        cf_cancer_volume = np.zeros((num_patients, num_cf_sequences, projection_horizon + 1))
        cf_chemo_dosage = np.zeros_like(cf_cancer_volume)
        cf_radio_dosage = np.zeros_like(cf_cancer_volume)
        if toxicity:
            cf_toxicity = np.zeros_like(cf_cancer_volume)

        for i in range(num_patients):
            alpha = alphas[i]
            beta = betas[i]
            beta_c = beta_cs[i]
            rho = rhos[i]
            K = Ks[i]

            if toxicity:
                kg = kgs[i]
                kl1 = kl1s[i]
                kl2 = kl2s[i]
                kl3 = kl3s[i]

            for start_t in counterfactual_start_times:
                if start_t + projection_horizon >= num_time_steps:
                    continue  # nicht genug Zeit nach hinten

                # Zustand am Startzeitpunkt speichern
                state_vol = cancer_volume[i, start_t]
                state_chemo = chemo_dosage[i, start_t]
                state_radio = radio_dosage[i, start_t]
                state_toxic = toxic[i, start_t] if toxicity else None

                for seq_idx, treatment_seq in enumerate(counterfactual_treatment_sequences):
                    # Initialisierung
                    cf_cancer_volume[i, seq_idx, 0] = state_vol
                    cf_chemo_dosage[i, seq_idx, 0] = state_chemo
                    cf_radio_dosage[i, seq_idx, 0] = state_radio
                    if toxicity:
                        cf_toxicity[i, seq_idx, 0] = state_toxic

                    for h in range(projection_horizon):
                        # Counterfaktische Dosen sind Arrays der Form (projection_horizon, 2)
                        current_chemo_dose = treatment_seq[h][0]
                        current_radio_dose = treatment_seq[h][1]

                        prev_chemo_dose = cf_chemo_dosage[i, seq_idx, h]
                        # Halbwertszeitberechnung
                        chemo_dose_updated = prev_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
                        cf_chemo_dosage[i, seq_idx, h + 1] = chemo_dose_updated
                        cf_radio_dosage[i, seq_idx, h + 1] = current_radio_dose

                        # Tumorvolumen update
                        prev_vol = cf_cancer_volume[i, seq_idx, h]
                        noise_val = 0  # Für Counterfaktisch kannst du Noise weglassen oder zufällig neu generieren
                        new_vol = prev_vol * (
                            1
                            + rho * np.log(K / prev_vol)
                            - beta_c * chemo_dose_updated
                            - (alpha * current_radio_dose + beta * current_radio_dose ** 2)
                            + noise_val
                        )
                        new_vol = np.clip(new_vol, 0, tumour_death_threshold)
                        cf_cancer_volume[i, seq_idx, h + 1] = new_vol

                        # Optional: Stoppen bei Tod oder Heilung in counterfaktischer Simulation

                        if toxicity:
                            prev_tox = cf_toxicity[i, seq_idx, h]
                            new_tox = prev_tox * (
                                1
                                + kg * prev_tox * (1 - prev_tox / initial_toxicity[i])
                                - kl1 * chemo_dose_updated
                                - kl2 * current_radio_dose
                                - kl3 * prev_vol
                                + 0  # kein Rauschen im Counterfaktischen hier
                            )
                            cf_toxicity[i, seq_idx, h + 1] = new_tox

        # # Counterfaktische Ergebnisse an Outputs anhängen
        # if toxicity:
        #     outputs["cancer_volume"] = cf_cancer_volume
        #     outputs["chemo_dosage"] = cf_chemo_dosage
        #     outputs["radio_dosage"] = cf_radio_dosage
        #     outputs["toxicity"] = cf_toxicity
        # else:
        #     outputs["cancer_volume"] = cf_cancer_volume
        #     outputs["chemo_dosage"] = cf_chemo_dosage
        #     outputs["radio_dosage"] = cf_radio_dosage
        
        # Outputs ohne counterfaktische Simulation
        if toxicity:
            outputs = {
                "cancer_volume": cancer_volume,
                "chemo_dosage": chemo_dosage,
                "radio_dosage": radio_dosage,
                "chemo_application": chemo_application_point,
                "radio_application": radio_application_point,
                "chemo_probabilities": chemo_probabilities,
                "radio_probabilities": radio_probabilities,
                "sequence_lengths": sequence_lengths,
                "patient_types": patient_types,
                "toxicity": toxic,
            }
            
            outputs["cancer_volume_cf"] = cf_cancer_volume
            outputs["chemo_dosage_cf"] = cf_chemo_dosage
            outputs["radio_dosage_cf"] = cf_radio_dosage
            outputs["toxicity_cf"] = cf_toxicity
        else:
            outputs = {
                "cancer_volume": cancer_volume,
                "chemo_dosage": chemo_dosage,
                "radio_dosage": radio_dosage,
                "chemo_application": chemo_application_point,
                "radio_application": radio_application_point,
                "chemo_probabilities": chemo_probabilities,
                "radio_probabilities": radio_probabilities,
                "sequence_lengths": sequence_lengths,
                "patient_types": patient_types,
            }
            
            outputs["cancer_volume_cf"] = cf_cancer_volume
            outputs["chemo_dosage_cf"] = cf_chemo_dosage
            outputs["radio_dosage_cf"] = cf_radio_dosage

    else:
        # Outputs ohne counterfaktische Simulation
        if toxicity:
            outputs = {
                "cancer_volume": cancer_volume,
                "chemo_dosage": chemo_dosage,
                "radio_dosage": radio_dosage,
                "chemo_application": chemo_application_point,
                "radio_application": radio_application_point,
                "chemo_probabilities": chemo_probabilities,
                "radio_probabilities": radio_probabilities,
                "sequence_lengths": sequence_lengths,
                "patient_types": patient_types,
                "toxicity": toxic,
            }
            
        else:
            outputs = {
                "cancer_volume": cancer_volume,
                "chemo_dosage": chemo_dosage,
                "radio_dosage": radio_dosage,
                "chemo_application": chemo_application_point,
                "radio_application": radio_application_point,
                "chemo_probabilities": chemo_probabilities,
                "radio_probabilities": radio_probabilities,
                "sequence_lengths": sequence_lengths,
                "patient_types": patient_types,
            }
            

    return outputs



# def simulate_counterfactuals_treatment_seq1(simulation_params, seq_length, projection_horizon, cf_seq_mode='sliding_treatment',toxicity=False,continuous_therapy=False):
#     """
#     Simulation of test trajectories to asses a subset of multiple-step ahead counterfactuals
#     :param simulation_params: Parameters of the simulation
#     :param seq_length: Maximum trajectory length (number of factual time-steps)
#     :param cf_seq_mode: Counterfactual sequence setting: sliding_treatment / random_trajectories
#     :return: simulated data dict with number of rows equal to num_patients * seq_length * 2 * projection_horizon
#     """

#     if cf_seq_mode == 'sliding_treatment':
#         chemo_arr = np.stack([np.eye(projection_horizon, dtype=int),
#                               np.zeros((projection_horizon, projection_horizon), dtype=int)], axis=-1)
#         radio_arr = np.stack([np.zeros((projection_horizon, projection_horizon), dtype=int),
#                               np.eye(projection_horizon, dtype=int)], axis=-1)
#         treatment_options = np.concatenate([chemo_arr, radio_arr])
#     elif cf_seq_mode == 'random_trajectories':
#         treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 2))
#     else:
#         raise NotImplementedError()

#     total_num_radio_treatments = 1
#     total_num_chemo_treatments = 1

#     radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
#     # radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
#     chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
#     chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

#     # sort this
#     chemo_idx = np.argsort(chemo_days)
#     chemo_amt = np.array(chemo_amt)[chemo_idx]
#     chemo_days = np.array(chemo_days)[chemo_idx]
    
#     if continuous_therapy:
#         chemo_amt=[3,5,7]
#         radio_amt=[1,2,3]

#     drug_half_life = 1  # one day half life for drugs

#     # Unpack simulation parameters
#     initial_stages = simulation_params['initial_stages']
#     initial_volumes = simulation_params['initial_volumes']
#     alphas = simulation_params['alpha']
#     rhos = simulation_params['rho']
#     betas = simulation_params['beta']
#     beta_cs = simulation_params['beta_c']
#     Ks = simulation_params['K']
#     patient_types = simulation_params['patient_types']
#     window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy
#     #lag = simulation_params['lag']

#     # Coefficients for treatment assignment probabilities
#     chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
#     radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
#     chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
#     radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

#     num_patients = initial_stages.shape[0]

#     num_test_points = len(treatment_options) * num_patients * seq_length

#     # Commence Simulation
#     cancer_volume = np.zeros((num_test_points, seq_length + projection_horizon))
#     chemo_application_point = np.zeros((num_test_points, seq_length + projection_horizon))
#     radio_application_point = np.zeros((num_test_points, seq_length + projection_horizon))
#     sequence_lengths = np.zeros(num_test_points)
#     patient_types_all_trajectories = np.zeros(num_test_points)
#     patient_ids_all_trajectories = np.zeros(num_test_points)
#     patient_current_t = np.zeros(num_test_points)
    
    
#     if toxicity:
#         initial_toxicity = simulation_params["initial_toxicity"]
#         kgs = simulation_params["kg"]
#         kl1s = simulation_params["kl1"]
#         kl2s = simulation_params["kl2"]
#         kl3s = simulation_params["kl3"]
#         toxic = np.zeros((num_patients, seq_length))
#         noise_terms_toxic = 0.0015 * np.random.randn(
#                 num_patients,
#                 seq_length,
#             )    

#     test_idx = 0

#     # Run actual simulation
#     for i in tqdm(range(num_patients), total=num_patients):

#         # if i % 200 == 0:
#         #     logging.info("Simulating patient {} of {}".format(i, num_patients))

#         noise = 0.01 * np.random.randn(seq_length + projection_horizon)  # 5% cell variability
#         recovery_rvs = np.random.rand(seq_length)

#         # initial values
#         factual_cancer_volume = np.zeros(seq_length)
#         factual_chemo_dosage = np.zeros(seq_length)
#         factual_radio_dosage = np.zeros(seq_length)
#         factual_chemo_application_point = np.zeros(seq_length)
#         factual_radio_application_point = np.zeros(seq_length)
#         factual_chemo_probabilities = np.zeros(seq_length)
#         factual_radio_probabilities = np.zeros(seq_length)

#         chemo_application_rvs = np.random.rand(seq_length)
#         radio_application_rvs = np.random.rand(seq_length)
        
        
#         if continuous_therapy:
#             chemo_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])
#             radio_dosage_rvs = np.random.uniform(low=-0.05,high=0.05,size=[num_patients, seq_length])

#         factual_cancer_volume[0] = initial_volumes[i]

#         alpha = alphas[i]
#         beta = betas[i]
#         beta_c = beta_cs[i]
#         rho = rhos[i]
#         K = Ks[i]
        
#         if toxicity:
#             noise_toxic = noise_terms_toxic[i]
#             toxic[i, 0] = initial_toxicity[i]
#             kg=kgs[i]
#             kl1=kl1s[i]
#             kl2=kl2s[i]
#             kl3=kl3s[i]

#         for t in range(0, seq_length - 1):

#             # Factual prev_treatments and outcomes
#             current_chemo_dose = 0.0
#             previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

#             # Action probabilities + death or recovery simulations
#             # if t >= lag:
#             #     cancer_volume_used = cancer_volume[i, max(t - window_size - lag, 0):max(t - lag + 1, 0)]
#             # else:
#             cancer_volume_used = np.zeros((1,))
#             #cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
#             cancer_diameter_used = np.array(
#                 [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
#             cancer_metric_used = cancer_diameter_used

#             # probabilities
#             radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
#             chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] * (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

#             factual_chemo_probabilities[t] = chemo_prob
#             factual_radio_probabilities[t] = radio_prob

#             # Action application
#             # if radio_application_rvs[t] < radio_prob:
#             #     factual_radio_application_point[t] = 1
#             #     factual_radio_dosage[t] = radio_amt[0]

#             # if chemo_application_rvs[t] < chemo_prob:
#             #     factual_chemo_application_point[t] = 1
#             #     current_chemo_dose = chemo_amt[0]
                
                
#             #Action application
#             if radio_application_rvs[t] < radio_prob:
#                 factual_radio_application_point[t] = 1
#                 if continuous_therapy:
#                     if radio_prob+radio_dosage_rvs[t]<0.25:
#                         factual_radio_dosage[t] = radio_amt[0]
#                     elif radio_prob+radio_dosage_rvs[t]>0.25 and radio_prob+radio_dosage_rvs[t]<0.5:
#                         factual_radio_dosage[t] = radio_amt[1]
#                     else:
#                         factual_radio_dosage[t] = radio_amt[2]
#                     radio_application_point[t] = factual_radio_dosage[i,t]
                        
#                 else:
#                     factual_radio_dosage[t] = radio_amt[0]

#             if chemo_application_rvs[t] < chemo_prob:
#                 factual_chemo_application_point[t] = 1
#                 #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
#                 if continuous_therapy:
#                     if chemo_prob+chemo_dosage_rvs[t]<0.25:
#                         current_chemo_dose = chemo_amt[0]
#                     elif chemo_prob+chemo_dosage_rvs[t]>0.25 and chemo_prob+chemo_dosage_rvs[t]<0.5:
#                         current_chemo_dose = chemo_amt[1]
#                     else:
#                         current_chemo_dose = chemo_amt[2]
#                     chemo_application_point[t] = current_chemo_dose
                        
#                 else:
#                     current_chemo_dose = chemo_amt[0]           
            
#             # if radio_application_rvs[i, t] < radio_prob:
#             #     factual_radio_application_point[t] = 1
#             #     if continuous_therapy:
#             #         if radio_prob+radio_dosage_rvs[i,t]<0.25:
#             #             factual_radio_dosage[i, t] = radio_amt[0]
#             #         elif radio_prob+radio_dosage_rvs[i,t]>0.25 and radio_prob+radio_dosage_rvs[i,t]<0.5:
#             #             factual_radio_dosage[i, t] = radio_amt[1]
#             #         else:
#             #             factual_radio_dosage[i, t] = radio_amt[2]
#             #         radio_application_point[i, t] = factual_radio_dosage[i,t]
                        
#             #     else:
#             #         factual_radio_dosage[i, t] = radio_amt[0]

#             # if chemo_application_rvs[i, t] < chemo_prob:
#             #     factual_chemo_application_point[i, t] = 1
#             #     #Grenzen zwischen 0.1 und 0.25, 0.5] mit 2,3,5,7
#             #     if continuous_therapy:
#             #         if chemo_prob+chemo_dosage_rvs[i,t]<0.25:
#             #             current_chemo_dose = chemo_amt[0]
#             #         elif chemo_prob+chemo_dosage_rvs[i,t]>0.25 and chemo_prob+chemo_dosage_rvs[i,t]<0.5:
#             #             current_chemo_dose = chemo_amt[1]
#             #         else:
#             #             current_chemo_dose = chemo_amt[2]
#             #         chemo_application_point[i, t] = current_chemo_dose
                        
#             #     else:
#             #         current_chemo_dose = chemo_amt[0]                

#             # Update chemo dosage
#             factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

#             # Factual prev_treatments and outcomes
#             factual_cancer_volume[t + 1] = factual_cancer_volume[t] * \
#                 (1 + rho * np.log(K / factual_cancer_volume[t]) - beta_c * factual_chemo_dosage[t] -
#                     (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2) + noise[t + 1])

#             factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, TUMOUR_DEATH_THRESHOLD)

#             if cf_seq_mode == 'random_trajectories':
#                 treatment_options = np.random.randint(0, 2, (projection_horizon * 2, projection_horizon, 2))

#             for treatment_option in treatment_options:

#                 counterfactual_cancer_volume = np.zeros(shape=(t + 1 + projection_horizon + 1))
#                 counterfactual_chemo_application_point = np.zeros(shape=(t + 1 + projection_horizon))
#                 counterfactual_radio_application_point = np.zeros(shape=(t + 1 + projection_horizon))
#                 counterfactual_chemo_dosage = np.zeros(shape=(t + 1 + projection_horizon))
#                 counterfactual_radio_dosage = np.zeros(shape=(t + 1 + projection_horizon))

#                 counterfactual_cancer_volume[:t + 2] = factual_cancer_volume[:t + 2]
#                 counterfactual_chemo_application_point[:t + 1] = factual_chemo_application_point[:t + 1]
#                 counterfactual_radio_application_point[:t + 1] = factual_radio_application_point[:t + 1]
#                 counterfactual_chemo_dosage[:t + 1] = factual_chemo_dosage[:t + 1]
#                 counterfactual_radio_dosage[:t + 1] = factual_radio_dosage[:t + 1]

#                 for projection_time in range(0, projection_horizon):

#                     current_t = t + 1 + projection_time
#                     previous_chemo_dose = counterfactual_chemo_dosage[current_t - 1]

#                     current_chemo_dose = 0.0
#                     counterfactual_radio_dosage[current_t] = 0.0
#                     if treatment_option[projection_time][0] == 1:
#                         counterfactual_chemo_application_point[current_t] = 1
#                         current_chemo_dose = chemo_amt[0]

#                     if treatment_option[projection_time][1] == 1:
#                         counterfactual_radio_application_point[current_t] = 1
#                         counterfactual_radio_dosage[current_t] = radio_amt[0]

#                     counterfactual_chemo_dosage[current_t] = previous_chemo_dose * np.exp(
#                         -np.log(2) / drug_half_life) + current_chemo_dose

#                     counterfactual_cancer_volume[current_t + 1] = counterfactual_cancer_volume[current_t] *\
#                         (1 + rho * np.log(K / (counterfactual_cancer_volume[current_t] + 1e-07) + 1e-07) -
#                          beta_c * counterfactual_chemo_dosage[current_t] -
#                          (alpha * counterfactual_radio_dosage[current_t] + beta * counterfactual_radio_dosage[current_t] ** 2) +
#                          noise[current_t + 1])

#                 if (np.isnan(counterfactual_cancer_volume).any()):
#                     continue

#                 cancer_volume[test_idx][:t + 1 + projection_horizon + 1] = counterfactual_cancer_volume
#                 chemo_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_chemo_application_point
#                 radio_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_radio_application_point
#                 patient_types_all_trajectories[test_idx] = patient_types[i]
#                 patient_ids_all_trajectories[test_idx] = i
#                 patient_current_t[test_idx] = t

#                 sequence_lengths[test_idx] = int(t) + projection_horizon + 1
#                 test_idx = test_idx + 1
                
                
#                 if toxicity:
#                     toxic[i, t + 1] = toxic[i, t] * (
#                         1
#                         + kg * toxic[i, t] *(1-toxic[i,t]/(initial_toxicity[i])) 
#                         - kl1 * counterfactual_chemo_dosage[i, t]
#                         - kl2 * counterfactual_radio_dosage[i, t]
#                         - kl3 * cancer_volume[i, t]
#                         + noise_toxic[t]
#                     )    

#             if (factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD) or \
#                     recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
#                 break


    
#     if toxicity:
        
#         outputs = {'cancer_volume': cancer_volume[:test_idx],
#                     'chemo_application': chemo_application_point[:test_idx],
#                     'radio_application': radio_application_point[:test_idx],
#                     'sequence_lengths': sequence_lengths[:test_idx],
#                     'patient_types': patient_types_all_trajectories[:test_idx],
#                     'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
#                     'patient_current_t': patient_current_t[:test_idx],
#                     "toxicity": toxic
#                     }
#     else:
#         outputs = {'cancer_volume': cancer_volume[:test_idx],
#                     'chemo_application': chemo_application_point[:test_idx],
#                     'radio_application': radio_application_point[:test_idx],
#                     'sequence_lengths': sequence_lengths[:test_idx],
#                     'patient_types': patient_types_all_trajectories[:test_idx],
#                     'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
#                     'patient_current_t': patient_current_t[:test_idx],
#                     }

#     #print("Call to simulate counterfactuals data")
    
#     ####

#     ####

#     return outputs

###### 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Run Simulation


#if __name__ == "__main__":
    # logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    # np.random.seed(100)

    # num_time_steps = 10  # 6 month followup
    # num_patients = 10
    
    # toxicity=True
    
    # simulation_params = get_confounding_params(
    #     num_patients,
    #     chemo_coeff=5.0,
    #     radio_coeff=5.0,
    #     toxicity=toxicity
    # )
    # simulation_params["window_size"] = 15

    # projection_horizon = 5
    # treatment_options = np.array(
    #     [
    #         [(1, 0), (0, 0), (0, 1), (0, 0), (0, 0)],
    #         [(0, 0), (1, 0), (0, 1), (0, 0), (0, 0)],
    #     ],
    # )

    # outputs = simulate(simulation_params, num_time_steps, toxicity=toxicity)

    # print(outputs["cancer_volume"][:10])
    # print(outputs["chemo_probabilities"][:10])
    # print(outputs["radio_probabilities"][:10])

    # print("finished")
    
    #a,b=simulate_test(num_time_steps=5,num_patients=1,chemo_options=[0,3],radio_options=[0,1.5],chemo_coeff=4,radio_coeff=4,window_size=15)
    
    #save_raw_datapath="/Users/jaschob/Desktop/Jenni_v2/data_model/out"
    #save_raw_datapath="C:/Users/wendland/Documents/GitHub/TE-CDE-main"
    #write_to_file(
   #     a,
   #     f"{save_raw_datapath}/10_patients_testdata_5timepoints.p",
   # )
    
    #save_raw_datapath="/Users/jaschob/Desktop/Jenni_v2/data_model/out"
    #save_raw_datapath="C:/Users/wendland/Documents/GitHub/TE-CDE-main"
    #write_to_file(
    #    b,
    #    f"{save_raw_datapath}/10_patients_testdata_5timepoints_normalsimulation.p",
    #)