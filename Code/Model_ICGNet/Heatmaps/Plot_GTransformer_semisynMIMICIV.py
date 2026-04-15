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



# ##############################################################################


dataset = 'semisynMimic4'
save_res0 = '/Users/jaschob/Desktop/SemiSyn/heatmap_syn_mimic4_outcome_1'


save_heat=True
invert = True
vmin=0
vmax = None
invert = True


with open(save_res0+'.pkl', 'rb') as handle:
    heat_data0 = pickle.load(handle)

heat_data0_mean = heat_data0['mean']
heat_data0_std = heat_data0['std']

with open(save_res0+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data0_mean, f)
with open(save_res0+'_std.pkl', "wb") as f:
    pickle.dump(heat_data0_std, f)
####
    
heat_max = 24
vmax=None   
    
 #No Skale   
res_dic0_mean = heatmap_pred(title =' Value 0 ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')     
res_dic0_std = heatmap_pred(title = ' Value',dataset=dataset,load_map=save_res0+'_std.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')   



 #No Skale   
vmax=1.0
res_dic0_mean = heatmap_pred(title =' Value 0 ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')    


vmax=0.8
res_dic0_mean = heatmap_pred(title =' Value 0 ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')    



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
    
