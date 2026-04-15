import torch
import numpy as np

import pickle

import matplotlib.pyplot as plt
from utils_observableNODE import heatmap_pred


num_seed_i = 0
torch.manual_seed(num_seed_i)
np.random.seed(num_seed_i)

####
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

####
gamma=8
# ##############################################################################

# Data Gamma
dataset = 'DoseAI'
save_res0 = '/Users/jaschob/Desktop/paper/plots_nicolas/Cancer_Toxicity_Heatmaps_new/heatmap_cancer_volume_'+str(gamma)
save_res1 = '/Users/jaschob/Desktop/paper/plots_nicolas/Cancer_Toxicity_Heatmaps_new/heatmap_toxicity_'+str(gamma)

with open(save_res0+'.0.pkl', 'rb') as handle:
    heat_data0 = pickle.load(handle)

heat_data0_mean = heat_data0['mean']
heat_data0_std = heat_data0['std']

with open(save_res0+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data0_mean, f)
with open(save_res0+'_std.pkl', "wb") as f:
    pickle.dump(heat_data0_std, f)
####

with open(save_res1+'.0.pkl', 'rb') as handle:
    heat_data1 = pickle.load(handle)

heat_data1_mean = heat_data1['mean']
heat_data1_std = heat_data1['std']

with open(save_res1+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data1_mean, f)
with open(save_res1+'_std.pkl', "wb") as f:
    pickle.dump(heat_data1_std, f)    

    
save_heat=False
    
 #No Skale   
res_dic0_mean = heatmap_pred(title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=None)
if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
res_dic0_std = heatmap_pred(title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0+'_std.pkl' ,vmin=0,vmax=None)
if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  

res_dic1_mean = heatmap_pred(title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'_mean.pkl' ,vmin=0,vmax=None)
if save_heat: plt.savefig(save_res1 +'_mean.png', bbox_inches='tight')     
res_dic1_std = heatmap_pred( title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'_std.pkl' ,vmin=0,vmax=None)
if save_heat: plt.savefig(save_res1 +'_std.png', bbox_inches='tight')      
    
            
 #mean skala 1
res_dic0_mean = heatmap_pred(title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=1)
if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
res_dic0_std = heatmap_pred(title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0+'_std.pkl' ,vmin=0,vmax=0.25)
if save_heat: plt.savefig(save_res0 +'_std2.png', bbox_inches='tight')  



res_dic1_mean = heatmap_pred(title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'_mean.pkl' ,vmin=0,vmax=1)
if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
res_dic1_std = heatmap_pred( title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'_std.pkl' ,vmin=0,vmax=0.25)
if save_heat: plt.savefig(save_res1 +'_std2.png', bbox_inches='tight')      
    
            



#Data2
#save_res0 = '/Users/jaschob/Desktop/Cancer_Toxicity_Heatmaps/heatmap_cancer_volume_'+str(gamma)
#save_res1 = '/Users/jaschob/Desktop/Cancer_Toxicity_Heatmaps/heatmap_toxicity_'+str(gamma)


        
#invert = True
# res_dic0_mean = heatmap_pred(title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'.0.pkl',vmin=vmin,vmax=1,invert=invert,  model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res0 +'_mean1.png', bbox_inches='tight')  
# #res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0 ,vmin=0,vmax=0.1)
# #if save_heat: plt.savefig(save_res0 +'_std1.png', bbox_inches='tight')  

# res_dic1_mean = heatmap_pred(title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'.0.pkl' ,vmin=vmin,vmax=1,invert=invert,  model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res1 +'_mean1.png', bbox_inches='tight')      
# #res_dic1_std = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 ,vmin=0,vmax=0.1)
# #if save_heat: plt.savefig(save_res1 +'_std1.png', bbox_inches='tight') 


# res_dic0_mean = heatmap_pred(title =str(gamma)+' Tumor Volum ',dataset=dataset,load_map=save_res0 +'.0.pkl',vmin=vmin,vmax=None,invert=invert,  model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res0 +'_mean.png', bbox_inches='tight')  
# #res_dic0_std = heatmap_pred(x_test,t_test,u_test, title = str(gamma)+ ' Tumor Volum ',dataset=dataset,load_map=save_res0 ,vmin=0,vmax=0.1)
# #if save_heat: plt.savefig(save_res0 +'_std.png', bbox_inches='tight')  

# res_dic1_mean = heatmap_pred(title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1+'.0.pkl' ,vmin=vmin,vmax=None,invert=invert,  model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res1 +'_mean.png', bbox_inches='tight')      
# #res_dic1_std = heatmap_pred(x_test,t_test,u_test, title =str(gamma)+' Weight ',dataset=dataset,load_map=save_res1 ,vmin=0,vmax=0.1)
# #if save_heat: plt.savefig(save_res1 +'_std.png', bbox_inches='tight') 

#print
for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,2)),np.flipud(np.round(res_dic0_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,2)),np.flipud(np.round(res_dic1_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))  


# ####
# for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,3)),np.flipud(np.round(res_dic0_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))    
    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,3)),np.flipud(np.round(res_dic1_std.T,3))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  
#         else:
#             m_str = f'{m:.3f}'
#             s_str = f'{s:.3f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))  




