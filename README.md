# ObsNODE
This GitHub repository provides the implementation of "Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time" by Wendland, Freitag, and Kschischo [1].
ObsNODE addresses the challenge of estimating causal effects in continuous-time decision-making settings, where system dynamics are often only partially observed. We demonstrate that recovering these latent dynamics from observed data—referred to as observability—is crucial for identifying how treatments influence outcomes over time.
Building on this insight, we introduce Observable Neural ODEs (ObsNODEs) a class of Neural ODE models specifically designed for causal forecasting under time-varying treatments. ObsNODEs enforce an observable structure that enables the reconstruction of latent states from past observations, thereby allowing for reliable prediction of outcomes under hypothetical treatment scenarios.
We evaluate our approach on synthetic cancer treatment simulations [2,3], semi synthetic MIMIC-IV data [4], and a real-world sepsis case study [5]. Across all settings, ObsNODEs demonstrate strong performance compared to baseline models, including OptAB [6], DoseAI [7], ICG-Net [8], and SCIP-Net [9].



# Contents of the Repository

* Scripts for running ObsNODE on synthetic cancer data and semi-synthetic MIMIC-IV and MIMIC-IV data, including detailed explanations, are located in the Code/Model_ObservableNODE folder. 
* Plots from the paper "Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time", along with the complete set of tables from the stability analysis, are provided in the Code/Plots folder.
*The implementation of the baseline models can be found in the Code/Model_(model_name) folder.
* Scripts for preprocessing the MIMIC-IV data, as well as for generating the semi-synthetic MIMIC-IV data and synthetic cancer data, are available in the Code/Data folder. 


# Package dependencies

* The package dependencies required to run the MIMIC-IV preprocessing scripts are located in the Code/Data/MIMIC-IV folder. 
* The Python package dependencies for running the ObsNODE model can be found in the Code/Model_observableNODE directory. 
* The package dependencies for running the baseline models (ObtAB, DoseAI, ICG-Netand SCIP-Net) are provided in their respective folders under Code/Model_(model_name).



# How to run ObsNODE

* To perform hyperparameter optimization, run the hypopt_ObservableNODE_(dataset).py script. 
* Train the ObsNODE model by executing training_ObservableNODE_seed_(dataset).py. 
* The compute_treatment_influences script generates predictions based on predefined treatment scenarios. 
* To reproduce the plots from the paper, run the plotting functions in the plot_ObservableNODE_(dataset).py script.
* Remark: Comprehensive descriptions of the code can be found in the utils_observableNODE.py file.






# References

[1] Wendland J., Freitag N. & Kschischo M., "Observable Neural ODEs for Identifiable Causal Forecasting in Continuous Time“

[2] Geng C., Paganett H. & Grassberger, C. "Prediction of treatment response for combined chemo-and radiation therapy for non-small cell lung cancer patients using a bio-mathematical model". In Scientific Reports, vol. 7.1, p.13542, ISSN: 0090-3493, DOI: 10.1038/s41598-017-13646-z (2017)

[3] Hadjiandrou M., & Mitsis G. "Mathematical Modeling of Tumor Growth, Drug-Resistance, Toxicity, and Optimal Therapy Design". In IEEE Transactions on Biomedical Engineering, vol. 61.2, pp. 415–425. ISSN: 0018-9294, 1558-2531. DOI: 10.1109/TBME.2013.2280189 (2014)

[4] Schulam P., Saria S., „Reliable Decision Support using Counterfactual Models“. In Neural Information Processing Systems (NIPS) (2017), DOI: 

https://doi.org/10.48550/arXiv.1703.10651


[5] Johnson A. E. W. et al. "MIMIC-IV, a freely accessible electronic health record dataset". Sci. Data 10, 1 (2023). URL https://www.nature.com/articles/s41597-022-01899-x

[6] Wendland P., Schenkel-Häger C., Wenningmann, I. & Kschischo, M. "An optimal antibiotic selection framework for Sepsis patients using Artificial Intelligence". In NPJ Digital Medicine, vol. 7.1, p.343. ISSN: 2398-6352. DOI: 10.1038/s41746-024-01350-y (2024)

[7] Wendland P. , Wendland J. & Kschischo M. "Counterfactual AI for Dynamic Dose Optimization with Side-Effect Constraints" DOI: https://doi.org/10.36227/techrxiv.174970492.29621159/v1 (2025)

[8] Hess K., Frauen D., Melnychuk V., Feuerriegel S., "IGC-Net for conditional average potential outcome estimation over time", in The Fourteenth International Conference on Learning Representations. (2026) DOI: 

https://doi.org/10.48550/arXiv.2405.21012


[9] Hess K., Feuerriegel S., "Stabilized Neural Prediction of Potential Outcomes" in Continuous Time. in The Thirteen International Conference on Learning Representations. (2025) DOI: 

https://doi.org/10.48550/arXiv.2405.21012






