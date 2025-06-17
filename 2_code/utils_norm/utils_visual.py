import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nibabel.freesurfer.io import read_annot
from nilearn import plotting, datasets

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
                    f.write("-" * 80 + "\n")
        break_down_p = df[['BrainArea', log_col]]
        break_down_p.loc[break_down_p[log_col] >= 0.05, log_col] = np.nan
        break_down_p.rename(columns={'BrainArea': 'eid'}, inplace=True)
        
        plot_brain_stat_figure_half(break_down_p, color_map, root_dir, results_dir, 
                            results_name+'_FDR', vmin=vmin, vmax=vmax, 
                            threshold=threshold, colorbar=True)
                
    break_down_p = df[['BrainArea', plot_col]].copy()
    break_down_p.loc[break_down_p[plot_col] >= 0.05, plot_col] = np.nan
    break_down_p.rename(columns={'BrainArea': 'eid'}, inplace=True)
    if break_down_p[plot_col].isna().all():
        return
    plot_brain_stat_figure_half(break_down_p, color_map, root_dir, results_dir, 
                          results_name, vmin=vmin, vmax=vmax, 
                          threshold=threshold, colorbar=True)

def plot_brain_stat_figure_half(deviation_percent_temp, cmap, docu_dir, results_dir, file_name, vmin=None, vmax=None, threshold=None, colorbar=True):
    """
    Plot brain statistics figure with hemisphere data.
    
    Parameters:
    -----------
    deviation_percent_temp : pd.DataFrame
        Data containing deviation percentages
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
    
    def prepare_data(deviation_percent_temp):
        deviation_percent_temp['eid'] = deviation_percent_temp['eid'].str.replace('\+', '_and_', regex=True)
        
        # Check if data has hemisphere information
        if deviation_percent_temp['eid'].str.contains('lh|rh').any():
            deviation_percent_temp[['temp1', 'temp2']] = deviation_percent_temp['eid'].str.split(pat="_", n=1, expand=True)
            deviation_percent_temp.temp1 = deviation_percent_temp.temp1.apply(lambda x: 'left' if 'lh' in x else x)
            deviation_percent_temp.temp1 = deviation_percent_temp.temp1.apply(lambda x: 'right' if 'rh' in x else x)
            deviation_percent_temp['hemisphere'] = deviation_percent_temp['temp1']
            deviation_percent_temp['ROI'] = deviation_percent_temp['temp2']
        else:
            # If no hemisphere info, treat all as left hemisphere
            deviation_percent_temp['hemisphere'] = 'left'
            deviation_percent_temp['ROI'] = deviation_percent_temp['eid']
            
        return deviation_percent_temp

    def get_parcellation_data(deviation_percent_temp, docu_dir):
        big_fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

        percentage_left = deviation_percent_temp[deviation_percent_temp.hemisphere == 'left']

        annotFile = os.path.join(docu_dir, 'lh.aparc.a2009s.annot')
        l_labels, _, nl_idx = read_annot(annotFile)
        l_labels = np.where(l_labels == -1, np.nan, l_labels)
        nl_idx = [x.decode() for x in nl_idx]

        nl_idx = pd.DataFrame(nl_idx, columns=['ROI'])
        nl_idx['ROI'] = nl_idx['ROI'].str.replace('-', '_')
        percentage_left['ROI'] = percentage_left['ROI'].str.replace('-', '_')
        nl_left = pd.merge(nl_idx, percentage_left[['ROI', 'data_to_plot']], on='ROI', how='left')
        nl_left = nl_left['data_to_plot'].to_numpy()

        a_list = list(range(1, 76))
        parcellation_temp_l = l_labels
        for j in a_list:
            parcellation_temp_l = np.where(parcellation_temp_l == j, nl_left[j], parcellation_temp_l)

        return big_fsaverage, parcellation_temp_l

    def plot_figure(big_fsaverage, parcellation_temp_l, results_dir, file_name, cmap, vmin, vmax, threshold, colorbar):
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

        # Save as SVG for high quality vector graphics
        plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.svg'), 
                   dpi=300, bbox_inches='tight', format='svg')
        print(f'brain_visualization_{file_name}.svg saved')
        
        # Also save as PNG for compatibility
        plt.savefig(os.path.join(results_dir, f'brain_visualization_{file_name}.png'), 
                   dpi=300, bbox_inches='tight')
        
        plt.clf()  # Clear the current figure
        plt.close('all')  # Close all figures
        gc.collect()  # Run garbage collection to free up memory

    deviation_percent_temp = prepare_data(deviation_percent_temp)

    data_columns = [col for col in deviation_percent_temp.columns if col not in ['eid', 'temp1', 'temp2', 'hemisphere', 'ROI']]
    os.makedirs(results_dir, exist_ok=True)
    for column in data_columns:
        deviation_percent_temp['data_to_plot'] = deviation_percent_temp[column]
        big_fsaverage, parcellation_temp_l = get_parcellation_data(deviation_percent_temp, docu_dir)
        plot_figure(big_fsaverage, parcellation_temp_l, results_dir, file_name, cmap, vmin, vmax, threshold, colorbar)
