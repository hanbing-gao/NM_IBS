import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils_norm.myblr import predict, warp_predictions
from pcntoolkit.util.utils import create_design_matrix
from nibabel.freesurfer.io import read_annot
from nilearn import plotting, datasets

def visual_norm_model(sex, idp, out_dir, cols_cov, site_ids, xmin, xmax, patient_data=None,save_path=None):

    if sex == 0: 
        clr = 'green'
    else:
        clr = 'blue'

    idp_dir = os.path.join(out_dir, idp)
    os.chdir(idp_dir)
    hyp = np.loadtxt(os.path.join(idp_dir, 'hyp.txt'))
    cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
    resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
    X_tr = np.loadtxt(cov_file_tr)
    Y_tr = np.loadtxt(resp_file_tr)

    # create dummy data for visualisation
    print('configuring dummy data ...')
    xx = np.arange(xmin, xmax, 0.5)
    X0_dummy = np.zeros((len(xx), 2))
    # age
    X0_dummy[:,0] = xx
    # sex
    X0_dummy[:,1] = sex
    # create the design matrix
    X_dummy = create_design_matrix(X0_dummy, xmin=xmin, xmax=xmax, site_ids=None, all_sites=site_ids)
    # save the dummy covariates
    cov_file_dummy = os.path.join(out_dir,'cov_bspline_dummy_mean_'+str(sex)+'.txt')
    np.savetxt(cov_file_dummy, X_dummy)
    print('save to:'+cov_file_dummy)
    sns.set(style='whitegrid')
    plt.figure(figsize=(10,5),dpi=300)
    parameters = {'axes.labelsize': 30, 'axes.titlesize': 40, 'xtick.labelsize':30,'ytick.labelsize':30,'legend.fontsize':30,'legend.title_fontsize':30}
    plt.rcParams.update(parameters)

    # load the true data points
    y_te = np.loadtxt(os.path.join(idp_dir, 'resp_te.txt'))
    X_te = np.loadtxt(os.path.join(idp_dir, 'cov_bspline_te.txt'))
    if patient_data is not None:
        # load patients data
        y_pa = np.loadtxt(os.path.join(idp_dir, 'resp_pat.txt'))
        X_pa = np.loadtxt(os.path.join(idp_dir, 'cov_bspline_pat.txt'))
    else:
        y_pa = None
        X_pa = None
    # set up the covariates for the dummy data
    yhat, s2 = predict(hyp, X_tr, Y_tr, X_dummy) 
    
    # get the warp and warp parameters
    warp_param = hyp[1:3]

    # then, we warp dummy predictions to create the plots
    med, pr_int = warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)

    # extract the different variance components to visualise
    beta = np.asarray([np.exp(hyp[0])]) 
    s2n = 1/beta # variation (aleatoric uncertainty)
    s2s = s2-s2n # modelling uncertainty (epistemic uncertainty)
    # plot the data points
    for sid, site in enumerate(site_ids):
        # plot the true test data points 
        # all data in the test set are present in the training set
        
        # first, we select the data points belonging to this particular site
        idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0))[0]
        if len(idx) == 0:
            print('No data for site', sid, site, 'skipping...')
            continue
        
        # then directly adjust the data
        idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
        y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
        
        # plot the (adjusted) data points
        plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color='blue', alpha = 0.5)
        
        # then directly adjust the data
        if patient_data is not None:
            # first, we select the data points belonging to this particular site
            idx = np.where(np.bitwise_and(X_pa[:,2] == sex, X_pa[:,sid+len(cols_cov)+1] !=0))[0]
            if len(idx) == 0:
                print('No data for site', sid, site, 'skipping...')
                continue
            idx_dummy = np.bitwise_and(X_dummy[:,1] > X_pa[idx,1].min(), X_dummy[:,1] < X_pa[idx,1].max())
            y_te_rescaled = y_pa[idx] - np.median(y_pa[idx]) + np.median(med[idx_dummy])
        
            # plot the (adjusted) data points
            plt.scatter(X_pa[idx,1], y_te_rescaled, s=4, color='red', alpha = 0.5)
        
    # plot the median of the dummy data
    plt.plot(xx, med, clr)

    # fill the gaps in between the centiles
    junk, pr_int25 = warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95 = warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99 = warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
    plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
    plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
    plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)
            
    # make the width of each centile proportional to the epistemic uncertainty
    junk, pr_int25l = warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95l = warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99l = warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
    junk, pr_int25u = warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
    junk, pr_int95u = warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
    junk, pr_int99u = warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])    
    plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
    plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)

    # plot actual centile lines
    plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
    plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)

    plt.xlabel('Age')
    # plt.ylabel(idp) 
    # plt.title(idp)
    plt.xlim((xmin,xmax))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path+'_'+str(sex), bbox_inches='tight')
        plt.close()

def plot_brain_mixedlm(df, plot_col, log_col, root_dir, results_dir, results_name, log_name, vmin=None, vmax=None, threshold=None, color_map='summer'):
    """
    Plot brain statistics and write breakdown analysis log.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing brain area data
    plot_col : str
        Column name for brain map to plot
    log_col : str 
        Column name for sig. FDR p-values to write to log
    root_dir : str
        Root directory path
    results_dir : str
        Directory to save results
    results_name : str
        Base name for result files
    log_name : str
        Name for log file
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    threshold : float, optional
        Threshold value for visualization
    color_map : str, optional
        Matplotlib colormap name, default 'summer'
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_file = os.path.join(results_dir, log_name+'.txt')

    break_down_fdr_p = df[['BrainArea', log_col]]
    break_down_fdr_p.loc[break_down_fdr_p[log_col] >= 0.05, log_col] = np.nan
    
    if break_down_fdr_p[log_col].isna().all():
        with open(log_file, 'a') as f:
            f.write('No significant results\n')
            f.write("-" * 80 + "\n")
    else:
        with open(log_file, 'a') as f:
            for index, row in break_down_fdr_p.iterrows():
                if row[log_col] < 0.05:
                    f.write(f"EID: {row['BrainArea']}, FDR p-value: {row[log_col]:.4f}\n")
                    print(f"{log_file}, EID: {row['BrainArea']}, FDR p-value: {row[log_col]:.4f}")
                    f.write("-" * 80 + "\n")
        break_down_p = df[['BrainArea', log_col]]
        break_down_p.loc[break_down_p[log_col] >= 0.05, log_col] = np.nan
        break_down_p.rename(columns={'BrainArea': 'eid'}, inplace=True)
        
        plot_brain_stat_figure(break_down_p, color_map, root_dir, results_dir, 
                            results_name+'_FDR', vmin=vmin, vmax=vmax, 
                            threshold=threshold, colorbar=True, half=False)
                
    
    df.loc[df['p'] >= 0.05, plot_col] = np.nan
    df.rename(columns={'BrainArea': 'eid'}, inplace=True)
    break_down_p = df[['eid', plot_col]].copy()
    if break_down_p[plot_col].isna().all():
        return
    plot_brain_stat_figure(break_down_p, color_map, root_dir, results_dir, 
                          results_name, vmin=vmin, vmax=vmax, 
                          threshold=threshold, colorbar=True, half=False)

def plot_brain_stat_figure(df, cmap, docu_dir, results_dir, file_name, vmin=None, vmax=None, threshold=None, colorbar=True, half=True):
    """
    Plot brain statistics figure with hemisphere data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data containing brain area data
    cmap : str
        Colormap name
    docu_dir : str
        Documentation directory path
    results_dir : str
        Directory to save results
    file_name : str
        Base name for output files
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    threshold : float, optional
        Threshold value for visualization
    colorbar : bool, optional
        Whether to include colorbar, default True
    half : bool, optional
        Whether to plot half brain, default True
    """
    def calculate_vmin_vmax(real_vmin, real_vmax):
        for i in range(4, -5, -1):
            temp_idx = (real_vmax - real_vmin) // (5 * 10 ** i)
            if temp_idx > 0:
                break
        temp_idx = 5 * 10 ** i
        # Make vmin and vmax a little bit bigger (absolute value) than the real min and max value
        vmin = np.floor(real_vmin / temp_idx) * temp_idx if real_vmin < 0 else np.floor(real_vmin / temp_idx) * temp_idx
        vmax = np.ceil(real_vmax / temp_idx) * temp_idx if real_vmax > 0 else np.ceil(real_vmax / temp_idx) * temp_idx
        return vmin, vmax
    
    def prepare_data(df):
        df['eid'] = df['eid'].str.replace('\+', '_and_', regex=True)
        
        # Check if data has hemisphere information
        if df['eid'].str.contains('lh|rh').any():
            df[['temp1', 'temp2']] = df['eid'].str.split(pat="_", n=1, expand=True)
            df.temp1 = df.temp1.apply(lambda x: 'left' if 'lh' in x else x)
            df.temp1 = df.temp1.apply(lambda x: 'right' if 'rh' in x else x)
            df['hemisphere'] = df['temp1']
            df['ROI'] = df['temp2']
        else:
            # If no hemisphere info, treat all as left hemisphere
            df['hemisphere'] = 'left'
            df['ROI'] = df['eid']
            
        return df

    def get_parcellation_data(df, docu_dir):
        big_fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

        df_left = df[df.hemisphere == 'left']
        df_right = df[df.hemisphere == 'right']

        annotFile = os.path.join(docu_dir, 'lh.aparc.a2009s.annot')
        l_labels, _, nl_idx = read_annot(annotFile)
        l_labels = np.where(l_labels == -1, np.nan, l_labels)
        annotFile = os.path.join(docu_dir, 'rh.aparc.a2009s.annot')
        r_labels, _, _ = read_annot(annotFile)
        r_labels = np.where(r_labels == -1, np.nan, r_labels)
        nl_idx = [x.decode() for x in nl_idx]

        nl_idx = pd.DataFrame(nl_idx, columns=['ROI'])
        nl_idx['ROI'] = nl_idx['ROI'].str.replace('-', '_')
        df_left['ROI'] = df_left['ROI'].str.replace('-', '_')
        nl_left = pd.merge(nl_idx, df_left[['ROI', 'data_to_plot']], on='ROI', how='left')
        df_right['ROI'] = df_right['ROI'].str.replace('-', '_')
        nl_right = pd.merge(nl_idx, df_right[['ROI', 'data_to_plot']], on='ROI', how='left')
        nl_left = nl_left['data_to_plot'].to_numpy()
        nl_right = nl_right['data_to_plot'].to_numpy()

        a_list = list(range(1, 76))
        parcellation_temp_l = l_labels
        for j in a_list:
            parcellation_temp_l = np.where(parcellation_temp_l == j, nl_left[j], parcellation_temp_l)
        parcellation_temp_r = r_labels
        for j in a_list:
            parcellation_temp_r = np.where(parcellation_temp_r == j, nl_right[j], parcellation_temp_r)

        return big_fsaverage, parcellation_temp_l, parcellation_temp_r

    def plot_figure(big_fsaverage, parcellation_temp_l, results_dir, file_name, cmap, vmin, vmax, threshold, colorbar, half=True, parcellation_temp_r=None):
        if half:
            if colorbar:
                fig, axs = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
            else:
                fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
            plt.subplots_adjust(wspace=0, hspace=0)
            cmap_temp = cmap
            if vmin is None or vmax is None:
                vmin, vmax = calculate_vmin_vmax(np.min(parcellation_temp_l), np.max(parcellation_temp_l))
            if colorbar:
                views = ["lateral", "medial", (270,270), 'dorsal', 'ventral']
            else:
                views = ["lateral", "medial", (270,270), 'dorsal']
            for i, view in enumerate(views):
                colorbar_temp = True if (colorbar and i == 4) else False
                mappable = plotting.plot_surf_stat_map(
                    big_fsaverage.infl_left,
                    parcellation_temp_l,
                    hemi="left",
                    view=view,
                    cmap=cmap_temp,
                    colorbar=colorbar_temp,
                    vmin=vmin,
                    vmax=vmax,
                    threshold=threshold,
                    bg_map=big_fsaverage.sulc_left,
                    figure=fig,
                    axes=axs[i]
                )

            # Save as PDF for high quality vector graphics
            plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.pdf'), 
                    dpi=300, bbox_inches='tight', format='pdf')
            print(f'brain_visualization_{file_name}.pdf saved')
            
            # Also save as PNG for compatibility
            plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.png'), 
                    dpi=300, bbox_inches='tight')
            
            plt.clf()  # Clear the current figure
            plt.close('all')  # Close all figures
            gc.collect()  # Run garbage collection to free up memory
        else:
            if colorbar:
                fig, axs = plt.subplots(1, 7, figsize=(15, 15), subplot_kw={'projection': '3d'})
            else:
                fig, axs = plt.subplots(3, 2, figsize=(10, 15), subplot_kw={'projection': '3d'})
            plt.subplots_adjust(wspace=0, hspace=0)
            cmap_temp = cmap
            if vmin is None or vmax is None:
                vmin, vmax = calculate_vmin_vmax(np.min([np.min(parcellation_temp_l), np.min(parcellation_temp_r)]), np.max([np.max(parcellation_temp_l), np.max(parcellation_temp_r)]))

            if colorbar:
                for i, (hemi, parcellation, view) in enumerate([
                    ("left", parcellation_temp_l, "lateral"),
                    ("left", parcellation_temp_l, "medial"),     
                    ("right", parcellation_temp_r, "medial"),
                    ("right", parcellation_temp_r, "lateral"),
                    ("left", parcellation_temp_l, (270,270)),
                    ("right", parcellation_temp_r, (270,270)),
                    ("right", parcellation_temp_r, "medial")
                ]):
                    if i == 6:
                        colorbar_temp = True
                    else:
                        colorbar_temp = False
                    mappable = plotting.plot_surf_stat_map(
                        getattr(big_fsaverage, f"infl_{hemi}"),
                        parcellation,
                        hemi=hemi,
                        view=view,
                        cmap=cmap_temp,
                        colorbar=colorbar_temp,
                        vmin=vmin,
                        vmax=vmax,
                        threshold=threshold,
                        bg_map=getattr(big_fsaverage, f"sulc_{hemi}"),
                        figure=fig,
                        axes=axs[i]
                    )
            else:
                colorbar_temp = False
                for i, (hemi, parcellation, view) in enumerate([
                    ("left", parcellation_temp_l, "lateral"),
                    ("right", parcellation_temp_r, "lateral"),
                    ("left", parcellation_temp_l, "medial"),
                    ("right", parcellation_temp_r, "medial"),
                    ("left", parcellation_temp_l, (270,270)),
                    ("right", parcellation_temp_r, (270,270)),
                ]):
                    mappable = plotting.plot_surf_stat_map(
                        getattr(big_fsaverage, f"infl_{hemi}"),
                        parcellation,
                        hemi=hemi,
                        view=view,
                        cmap=cmap_temp,
                        colorbar=colorbar_temp,
                        vmin=vmin,
                        vmax=vmax,
                        threshold=threshold,
                        bg_map=getattr(big_fsaverage, f"sulc_{hemi}"),
                        figure=fig,
                        axes=axs[i // 2, i % 2]
                    )
            plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.png'), 
                        dpi=300, bbox_inches='tight')
            # plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.pdf'), 
            #             dpi=300, bbox_inches='tight', format='pdf')
            plt.clf()  # Clear the current figure
            plt.close('all')  # Close all figures
            gc.collect()  # Run garbage collection to free up memory
    df = prepare_data(df)

    data_columns = [col for col in df.columns if col not in ['eid', 'temp1', 'temp2', 'hemisphere', 'ROI']]
    os.makedirs(results_dir, exist_ok=True)
    for column in data_columns:
        df['data_to_plot'] = df[column]
        big_fsaverage, parcellation_temp_l, parcellation_temp_r = get_parcellation_data(df, docu_dir)
        if half:
            parcellation_temp_r = None
        plot_figure(big_fsaverage, parcellation_temp_l, results_dir, file_name, cmap, vmin, vmax, threshold, colorbar, half, parcellation_temp_r)
