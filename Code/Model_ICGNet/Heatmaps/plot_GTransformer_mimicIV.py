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


dataset = 'mimic4'
save_res0 = '/Users/jaschob/Desktop/MIMICIV_new/heatmap_SOFA'
save_res1 = '/Users/jaschob/Desktop/MIMICIV_new/heatmap_creatinine'
save_res2 = '/Users/jaschob/Desktop/MIMICIV_new/heatmap_bilirubin'
save_res3 = '/Users/jaschob/Desktop/MIMICIV_new/heatmap_alt'
    

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

with open(save_res1+'.pkl', 'rb') as handle:
    heat_data1 = pickle.load(handle)

heat_data1_mean = heat_data1['mean']
heat_data1_std = heat_data1['std']

with open(save_res1+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data1_mean, f)
with open(save_res1+'_std.pkl', "wb") as f:
    pickle.dump(heat_data1_std, f)        
    
####
with open(save_res2+'.pkl', 'rb') as handle:
    heat_data2 = pickle.load(handle)

heat_data2_mean = heat_data2['mean']
heat_data2_std = heat_data2['std']

with open(save_res2+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data2_mean, f)
with open(save_res2+'_std.pkl', "wb") as f:
    pickle.dump(heat_data2_std, f)
###

with open(save_res3+'.pkl', 'rb') as handle:
    heat_data3= pickle.load(handle)

heat_data3_mean = heat_data3['mean']
heat_data3_std = heat_data3['std']

with open(save_res3+'_mean.pkl', "wb") as f:
    pickle.dump(heat_data3_mean, f)
with open(save_res3+'_std.pkl', "wb") as f:
    pickle.dump(heat_data3_std, f)    
            
    
heat_max = 12
vmax=None   
    
 #No Skale   
res_dic0_mean = heatmap_pred(title =' SOFA ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')     
res_dic0_std = heatmap_pred(title = ' SOFA',dataset=dataset,load_map=save_res0+'_std.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')   

res_dic1_mean = heatmap_pred(title = ' Creatinine ',dataset=dataset,load_map=save_res1+'_mean.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res1 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')     
res_dic1_std = heatmap_pred( title = ' Creatinine ',dataset=dataset,load_map=save_res1+'_std.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res1 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')   

res_dic2_mean = heatmap_pred(title = ' Bilirubin total ',dataset=dataset,load_map=save_res2+'_mean.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res2 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')      
res_dic2_std = heatmap_pred( title = ' Bilirubin total ',dataset=dataset,load_map=save_res2+'_std.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res2 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')   

res_dic3_mean = heatmap_pred(title = ' ALT ',dataset=dataset,load_map=save_res3+'_mean.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res3 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')     
res_dic3_std = heatmap_pred( title = ' ALT ',dataset=dataset,load_map=save_res3+'_std.pkl' ,vmin=0,vmax=None, heat_max=heat_max)
if save_heat: plt.savefig(save_res3 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')    



 #No Skale   
vmax=1.0
res_dic0_mean = heatmap_pred(title =' SOFA ',dataset=dataset,load_map=save_res0 +'_mean.pkl',vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')    

res_dic1_mean = heatmap_pred(title = ' Creatinine ',dataset=dataset,load_map=save_res1+'_mean.pkl' ,vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res1 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')   

res_dic2_mean = heatmap_pred(title = ' Bilirubin total ',dataset=dataset,load_map=save_res2+'_mean.pkl' ,vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res2 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')    

res_dic3_mean = heatmap_pred(title = ' ALT ',dataset=dataset,load_map=save_res3+'_mean.pkl' ,vmin=0,vmax=vmax, heat_max=heat_max)
if save_heat: plt.savefig(save_res3 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')  


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
    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic2_mean.T,2)),np.flipud(np.round(res_dic2_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('')  
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))    
    
for mean_row, std_row in zip(np.flipud(np.round(res_dic3_mean.T,2)),np.flipud(np.round(res_dic3_std.T,2))):
    line = []
    for m, s in zip(mean_row, std_row):
        if np.isnan(m):
            line.append('') 
        else:
            m_str = f'{m:.2f}'
            s_str = f'{s:.2f}'
            line.append(f'{m_str}±{s_str} | ')
    print(' '.join(line))   
    


# heat_max =12
# vmin=0
# vmax=None
# save_heat=True
# res_dic0_mean = heatmap_pred( title =' SOFA ',dataset=dataset,load_map=save_res0 +'.pkl',heat_max=heat_max,vmin=vmin,vmax=vmax,invert=invert, model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res0 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')  
# # res_dic0_std = heatmap_pred( title = ' SOFA ',dataset=dataset,load_map=save_res0 +'_std.pkl',heat_max=heat_max, model_name = 'GTransfomer')
# # if save_heat: plt.savefig(save_res0 +'_std_'+str(heat_max)+'.png', bbox_inches='tight')  

# res_dic1_mean = heatmap_pred( title =' Creatinine ',dataset=dataset,load_map=save_res1 +'.pkl',heat_max=heat_max,vmin=vmin,vmax=vmax,invert=invert, model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res1 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')      
# # res_dic1_std = heatmap_pred(title =' Creatinine ',dataset=dataset,load_map=save_res1 +'_std.pkl',heat_max=heat_max, model_name = 'GTransfomer')
# # if save_heat: plt.savefig(save_res1 +'_std_'+str(heat_max)+'.png', bbox_inches='tight') 

# res_dic2_mean = heatmap_pred(title =' Bilirubin total ',dataset=dataset,load_map=save_res2 +'.pkl',heat_max=heat_max,vmin=vmin,vmax=vmax,invert=invert, model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res2 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')      
# # res_dic2_std = heatmap_pred( title =' Bilirubin total ',dataset=dataset,load_map=save_res2 +'_std.pkl',heat_max=heat_max, model_name = 'GTransfomer')
# # if save_heat: plt.savefig(save_res2 +'_std_'+str(heat_max)+'.png', bbox_inches='tight') 

# res_dic3_mean = heatmap_pred(title =' ALT ',dataset=dataset,load_map=save_res3 +'.pkl',heat_max=heat_max,vmin=vmin,vmax=vmax,invert=invert, model_name = 'GTransfomer')
# if save_heat: plt.savefig(save_res3 +'_mean_'+str(heat_max)+'_vmax_'+str(vmax)+'.png', bbox_inches='tight')      
# # res_dic3_std = heatmap_pred( title =' ALT ',dataset=dataset,load_map=save_res3 +'_std.pkl',heat_max=heat_max, model_name = 'GTransfomer')
# # if save_heat: plt.savefig(save_res3 +'_std_'+str(heat_max)+'.png', bbox_inches='tight') 


# for mean_row, std_row in zip(np.flipud(np.round(res_dic0_mean.T,2)),np.flipud(np.round(res_dic0_std.T,2))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('') 
#         else:
#             m_str = f'{m:.2f}'
#             s_str = f'{s:.2f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))    
    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic1_mean.T,2)),np.flipud(np.round(res_dic1_std.T,2))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('') 
#         else:
#             m_str = f'{m:.2f}'
#             s_str = f'{s:.2f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line)) 
    
    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic2_mean.T,2)),np.flipud(np.round(res_dic2_std.T,2))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('')  
#         else:
#             m_str = f'{m:.2f}'
#             s_str = f'{s:.2f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))    
    
# for mean_row, std_row in zip(np.flipud(np.round(res_dic3_mean.T,2)),np.flipud(np.round(res_dic3_std.T,2))):
#     line = []
#     for m, s in zip(mean_row, std_row):
#         if np.isnan(m):
#             line.append('') 
#         else:
#             m_str = f'{m:.2f}'
#             s_str = f'{s:.2f}'
#             line.append(f'{m_str}±{s_str} | ')
#     print(' '.join(line))   
    