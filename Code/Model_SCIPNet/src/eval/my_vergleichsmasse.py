import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import sys
#from train_method_server_mimic4_ import predict

###


# def my_compute_prediction(x_test,t_test,u_test, index_t_s_i,model,n_x_dach=2, w=0,step_size=1,time_obs_pred=False,method='rk4',st=False,x_train=None,
#                           unscaled=False, step=None, variables_std=None,variables_mean=None,device = 'cpu',
#                           offset =None, max_horizon = None):
    
def my_compute_prediction(x_pred_i, x_true, index_t_s_i, unscaled=False, step=None, variables_std=None, variables_mean=None,device = 'cpu',
                          offset =None, max_horizon = None):    
    
    # Function to compute prediction measures: mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape
    
    # Inputs: 
    # x_pred: tensor from predicted values  #! Shape: (timepoints x number of patients x number of variables)?
    # x_true: tensor frome the ture values  #! Shape: (timepoints x number of patients x number of variables)?
    # step: Corresponds to number of timestamps aggregated for computing the measures, default: None
    # unscaled: Inficator, whether variables should be normalized or not, default False
    # variables_std: list of the standard deviations of the variables, default None
    # variables_mean: list of the means of the variables, default None
    # device: device of the model
    
    # Outputs: 
    # mse: Mean Square Errors, size: number of patients x timepoints x number of variables
    # rmse: Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_sd: Normalized (standard deviation) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_mean: Normalized (mean) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # nrmse_iqr: Normalized (inter-quartile range) Root Mean Square Errors, size: number of patients x timepoints x number of variables
    # mape: Mean Absolute Percentage Error, size: number of patients x timepoints x number of variables
    # mae: Mean Absolute Error, size: number of patients x timepoints x number of variables
    # wape: Weighted Absolute Percentage Error, size: number of patients x timepoints x number of variables
        

    # t_test_before, t_test_after     = t_test[:index_t_s_i+1].detach().clone(), t_test[index_t_s_i:].detach().clone()    
    x_true_before, x_true_after     = x_true.transpose(1,0)[:index_t_s_i+1].detach().clone(), x_true.transpose(1,0)[index_t_s_i:].detach().clone()
    # u_test_before, u_test_after     = u_test[:index_t_s_i+1].detach().clone(), u_test[index_t_s_i:].detach().clone()


    # x_pred_i, _, _,_ = predict(model=model,
    #                                                   step_size=step_size,
    #                                                   time_obs_pred=time_obs_pred,
    #                                                   t_before=t_test_before,
    #                                                   x_before=x_test_before,
    #                                                   t_after=t_test_after,
    #                                                   x_after=x_test_after,
    #                                                   method=method,
    #                                                   u_before=u_test_before,
    #                                                   index_t_s=index_t_s_i,
    #                                                   n_x_dach=n_x_dach,
    #                                                   w=w)

    # if st == True:
    #     for v_i in range(x_test.size(-1)):
    #         x_pred_i[..., v_i] = (x_pred_i[..., v_i]-np.nanmean(x_train[..., v_i]))/np.nanstd(x_train[..., v_i])
    #         x_test_after[..., v_i] = (x_test_after[..., v_i]-np.nanmean(x_train[..., v_i]))/np.nanstd(x_train[..., v_i])
    
    X=x_true_after.transpose(1,0)
    pred_output_val = x_pred_i
    
        #print(0)
    
    if unscaled:
        for v in range(pred_output_val.size(-1)):
            pred_output_val[:,:,v] = pred_output_val[:,:,v]*variables_std[v]+variables_mean[v]
        for v in range(X.size(-1)):
            X[:,:,v] = X[:,:,v]*variables_std[v]+variables_mean[v]    
            
    ### MSE, MAPE, WAPE, MAE ###
    mseloss = torch.nn.MSELoss()
    mse = torch.empty(size=pred_output_val.shape[1:],device=device)#torch.empty(size=pred_output_val.shape[1:],device=device)
    mse[:] = np.nan
    
    mapeloss=torchmetrics.MeanAbsolutePercentageError().to(device)
    mape = torch.empty(size=pred_output_val.shape[1:],device=device)
    mape[:]=np.nan
    
    #! torchmetric version compatibility
    # wapeloss=torchmetrics.WeightedMeanAbsolutePercentageError().to(device)
    # wape = torch.empty(size=pred_output_val.shape[1:],device=device)
    # wape[:]=np.nan 
    
    maeloss=torchmetrics.MeanAbsoluteError().to(device)
    mae = torch.empty(size=pred_output_val.shape[1:],device=device)
    mae[:]=np.nan
    
    if step is None:
        for i in range(mse.shape[0]):
            for j in range(mse.shape[1]):
                #offset wie ts
                # if offset != None:
                #     index_nan = (~X[:,offset+i,j:j+1].isnan())
                # else:    
                #     index_nan = (~X[:,i,j:j+1].isnan())
                index_nan = (~X[:,i,j:j+1].isnan())
                #print(pred_output_val[:,i,j:j+1][index_nan])
                #print( X[:,i,j:j+1][index_nan])
                
                mse[i,j]    = mseloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                mape[i,j]   = mapeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                #! torchmetric version compatibility
                #wape[i,j]   = wapeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                mae[i,j]    = maeloss(pred_output_val[:,i,j:j+1][index_nan], X[:,i,j:j+1][index_nan])
                #print(1)
    else:
        for i in range(mse.shape[0]):#step
            for j in range(mse.shape[1]):
                # if offset != None:
                #     index_nan = (~X[:,offset+i+1:min(offset+1+i+step,offset+1+mse.shape[0]),j:j+1].isnan())
                # else:
                #     index_nan = (~X[:,i:i+step,j:j+1].isnan())
                index_nan = (~X[:,i:i+step,j:j+1].isnan())    
                mse[i,j]    = mseloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                mape[i,j]   = mapeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                #! torchmetric version compatibility
                # wape[i,j]   = wapeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                mae[i,j]    = maeloss(pred_output_val[:,i:i+step,j:j+1][index_nan], X[:,i:i+step,j:j+1][index_nan])
                #print(2)
    
    ### RMSE ### 
    rmse=torch.sqrt(mse)
    
    X_std=X.clone()
    
    nrmse_sd = torch.empty(size=pred_output_val.shape,device=device)
    nrmse_sd[:] = np.nan
    
    nrmse_mean = torch.empty(size=pred_output_val.shape,device=device)
    nrmse_mean[:] = np.nan
    
    nrmse_iqr = torch.empty(size=pred_output_val.shape,device=device)
    nrmse_iqr[:] = np.nan
    
    for j in range(X.shape[2]):
        X_std[:,:,j][(X[:,:,j].isnan())] = np.nan
    
    if step is None:
        if offset != None:
            if max_horizon != None:
                X_sdt = X_std[:,offset+1:offset+max_horizon+1,:]
                
        dat_sd = torch.from_numpy(np.nanstd(X_std.cpu(),axis=0)).to(device)
        nrmse_sd = rmse/dat_sd
        #! nrmse_sd = rmse/dat_sd[1:]

        dat_mean = torch.from_numpy(np.nanmean(X_std.cpu(),axis=0)).to(device)
        nrmse_mean = rmse/dat_mean
        #! nrmse_mean = rmse/dat_mean[1:]

        dat_iqr = torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.75,axis=0)).to(device) - torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.25,axis=0)).to(device)
        nrmse_iqr=rmse/dat_iqr
        #! nrmse_iqr=rmse/dat_iqr[1:]
    else:
        for i in range(mse.shape[0]):#.step
            # if offset != None:
            #     X_std_i = X_std[:,offset+1+i:offset+1+i+step,:]
            # else:
            #     X_std_i = X_std[:,i:i+step,:]
            X_std_i = X_std[:,i:i+step,:]           
            dat_sd = torch.from_numpy(np.nanstd(X_std_i.cpu(),axis=(0,1))).to(device)
            dat_mean = torch.from_numpy(np.nanmean(X_std_i.cpu(),axis=(0,1))).to(device)
            dat_iqr = torch.from_numpy(np.nanquantile(X_std_i.cpu(),q=0.75,axis=(0,1))).to(device) - torch.from_numpy(np.nanquantile(X_std.cpu(),q=0.25,axis=(0,1))).to(device)
       
            nrmse_sd[i] = rmse[i]/dat_sd
            nrmse_mean[i] = rmse[i]/dat_mean
            nrmse_iqr[i]=rmse[i]/dat_iqr

    print(rmse.shape)  
    print(rmse)

    return mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, #! wape



# def heatmap_pred(x_test,t_test,u_test, list_index_t_s_pred,model,n_x_dach=2, w=0,step_size=1,time_obs_pred=False,method='rk4',st=False,x_train=False,
#                  index=0, offset=0, max_horizon=10,loss='rmse', rectilinear_index=None, step=None, unscaled=False, variables_std=None,variables_mean=None,variables=None,title=str(),save_link=None,save_map=None,load_map=None,vmin=None,vmax=None,colorbar=True,diagmultistep=False, dec_expand=False, med_dec=True, med_dec_start=False, invert=False):

def heatmap_pred(x_pred_list, x_true, list_index_t_s_pred, step_size=1,time_obs_pred=False,method='rk4',st=False,x_train=False,
                  index=0, offset=0, max_horizon=10,loss='rmse', rectilinear_index=None, step=None, unscaled=False, variables_std=None,variables_mean=None,variables=None,title=str(),save_link=None,save_map=None,load_map=None,vmin=None,vmax=None,colorbar=True,diagmultistep=False, dec_expand=False, med_dec=True, med_dec_start=False, invert=False):    
    #x_pred, x_true,unscaled=False, step=None, variables_std=None,variables_mean=None,device = 'cpu',offset =None, max_horizon = None for my_compute_prediction
    # Function to create a heatmap based on the prediction measures of the Decoder
    
    # model: Neural CDE model of the Encoder
    # model_decoder: Neural CDE model of the Decoder
    # validation_output: Tensor of output/treatment success, size: number of patients x timepoints (in hours) x 1 (dim of output)
    # validation_toxic: Tensor of the side effects, size: number of patients x timepoints (in hours) x number of side effects
    # validation_treatments: Tensor of the treatments, size: number of patients x timepoints (in hours), x number of treatments
    # covariables: Tensor of the covariables, size: number of patients x timepoints (in hours) x number of covariables (important: rectilinear_index has to correspond to the time dimension)
    # active_entries: Boolean Tensor indicating, whether patients is at ICU or discharged (training data), size: number of patients x timepoints (in hours) x 1
    # static: Tensor of static variables: number of patients x number of variables, default: None
    
    # index: Index of variable
    # offset: Corresponds to start timepoint for computation of prediction measures
    # max_horizon: Corresponds to end timepoint for computation of prediction measures
    # loss: Indicating the used loss
    # rectilinear_index: Time index of covariables tensor, default: None
    # step: Corresponds to number of timestamps aggregated for computing the measures, default: None
    # unscaled: Inficator, whether variables should be normalized or not, default False
    # variables_std: list of the standard deviations of the variables, default None
    # variables_mean: list of the means of the variables, default None
    # variables: list of all variables
    
    # title: Title of the plot, default: str() empty string
    # save_link: Link to save computed figures, default: None
    # save_map: Link to save computed prediction measures, default: None
    # load_map: Link to load computed map, default: None
    # vmin: Minimum value of heatmap, default: None (using minimum observed datapoint)
    # vmax: Maximum value of heatmap, default: None (using maximum observed datapoint)
    # colorbar: Indicator, whether to plot colorbar or not default: True
    # diagmultistep: Indicator, whether 1-step diagonal is plotted or not default: False
    
    # dec_expand Indicating whether initialization of decoder is expanded (at least with static data) or not default: False
    # sofa_expand: Indicating whether initialization of decoder is expanded by last measured sofa_score default: False
    # med_dec: Indicating, whether Treatments are used as control or not default: False
    # med_dec_start: Indicating, whether initialization of decoder is expanded by the last measured treatment default: True
    
    
    # Loading data (result data in diagonal form)
    if load_map is not None:
        with open(load_map, 'rb') as handle:
            data2 = pickle.load(handle)
    
    # Computing prediction measured
    else:
        heat_data_list = [np.full((max_horizon, max_horizon), np.nan) for _ in range(len(x_pred_list))]

        # if step is None or diagmultistep:
        #     #mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = prediction_measures(model, data_map, validation_output, validation_toxic, validation_treatments, covariables.clone(), active_entries, static, unscaled=unscaled, rectilinear_index=rectilinear_index,variables_std=variables_std,variables_mean=variables_mean,variables=variables)
            
        #     #mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = my_compute_prediction(x_pred, x_true)#without offset, 
        #     index_t_s_i = list_index_t_s_pred[0]
        #     mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = my_compute_prediction(x_test=x_test,
        #                                                                                         t_test=t_test,
        #                                                                                         u_test=u_test,
        #                                                                                         index_t_s_i=index_t_s_i,
        #                                                                                         model=model,
        #                                                                                         n_x_dach=n_x_dach,
        #                                                                                         w=w,step_size=step_size,
        #                                                                                         time_obs_pred=time_obs_pred,
        #                                                                                         method=method,
        #                                                                                         step = step,st=st,x_train=x_train)
            
        #     #define diagonal matrix
        #     #print(mse)
        #     if loss=='rmse':
        #         np.fill_diagonal(heat_data,rmse[offset:offset+max_horizon,index].cpu().detach().numpy())# offset default 1? 
        #     elif loss=='mse':
        #         np.fill_diagonal(heat_data,mse[offset:offset+max_horizon,index].cpu().detach().numpy())
        #     elif loss=='mae':
        #         np.fill_diagonal(heat_data,mae[offset:offset+max_horizon,index].cpu().detach().numpy())
        #     elif loss=='wape':
        #         np.fill_diagonal(heat_data,wape[offset:offset+max_horizon,index].cpu().detach().numpy())
        #     elif loss=='nrmse':
        #         np.fill_diagonal(heat_data,nrmse_sd[offset:offset+max_horizon,index].cpu().detach().numpy())
        
        for i in range(max_horizon):#-1
            #mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae, wape = prediction_measures_decoder(model,model_decoder, data_map,offset=offset,max_horizon=max_horizon,unscaled=unscaled, validation_output=validation_output, validation_toxic=validation_toxic, validation_treatments=validation_treatments, covariables=covariables.clone(), time_covariates=time_covariates, active_entries=active_entries, static=static, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=dec_expand, med_dec=med_dec, med_dec_start=med_dec_start)
            #print(i)

            for run_index, x_pred in enumerate(x_pred_list):
                index_t_s_i = list_index_t_s_pred[i]
                mse, rmse, nrmse_sd, nrmse_mean, nrmse_iqr, mape, mae = my_compute_prediction(x_pred[i], x_true[i], index_t_s_i)#,offset=offset, step=step,max_horizon=max_horizon)#with offset and more features... decoder func


                if loss=='rmse':
                    heat_data_list[run_index][i, i+1:] = rmse[:-1, index].cpu().detach().numpy()
                elif loss=='mse':
                    heat_data_list[run_index][i, i+1:] = mse[:-1, index].cpu().detach().numpy()
                elif loss=='mae':
                    heat_data_list[run_index][i, i+1:] = mae[:-1, index].cpu().detach().numpy()
                elif loss=='nrmse':
                    heat_data_list[run_index][i, i+1:] = nrmse_sd[:-1, index].cpu().detach().numpy()
       
        heat_data_np = np.array(heat_data_list)
        heat_data_mean = np.nanmean(heat_data_np, axis=0)
        heat_data_std = np.nanstd(heat_data_np, axis=0)

        if save_map is not None:
            with open(save_map, 'wb') as handle:
                pickle.dump({'mean': heat_data_mean, 'std': heat_data_std}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        heat_data = heat_data_mean
        data = np.ma.masked_invalid(heat_data)
        data2=ffill(data)
    
    if not invert:
        data2 = np.transpose(data2)
    
    # Creating plots
    if colorbar:
        fig, ax = plt.subplots(figsize=(5,4),dpi=400)
    else:
        fig, ax = plt.subplots(figsize=(4,4),dpi=400)
    
    if vmin is None:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=0, vmax=np.nanmax(data))
    else:
        heatmap = ax.pcolor(data2, cmap=plt.cm.seismic, 
                            vmin=vmin, vmax=vmax)
        
    # https://stackoverflow.com/a/16125413/190597 (Joe Kington)
    if colorbar:
        fig.colorbar(heatmap)
    
    # Setting some plotting options
    
    ax.set_xticks(np.arange(data2.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data2.shape[0])+0.5, minor=False)
    
    if max_horizon>72:
        a=np.arange(0,data2.shape[0],10)
        b=range(0,data2.shape[1],10)
    else:
        a=np.arange(0,data2.shape[0]+1,5)
        b=range(0,data2.shape[1]+1,5)
        
    ax.xaxis.set(ticks=a, ticklabels=a)

    ax.yaxis.set(ticks=b,ticklabels=b)
    
    if invert:
        ax.set_ylabel('Observed until treatment cycle x in months',fontsize= 13)
        ax.set_xlabel('Forecast horizon in treatment cycles',fontsize= 13)
    
    else:
        ax.set_xlabel('Observed until treatment cycle y in months',fontsize= 13)
        ax.set_ylabel('Forecast horizon in treatment cycles',fontsize= 13)
    
    
    if loss=='rmse':
        ax.set_title(title+'Root Mean Square Error')
    elif loss=='mse':
        ax.set_title(title+'Mean Square Error',fontsize= 13)
    elif loss=='mae':
        ax.set_title(title+'Mean Absolute Error')
    elif loss=='wape':
        ax.set_title(title+'WAPE')
    elif loss=='nrmse':
        ax.set_title(title+'NRMSE')
    
    if save_link is not None:
        plt.savefig(save_link)
    # if save_map is not None:
    #     with open(save_map, 'wb') as handle:
    #         pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out
