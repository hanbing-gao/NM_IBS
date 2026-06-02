# Main pipeline for normative modeling using BLR with warping.
# Adapted from PCNtoolkit (https://github.com/amarquand/PCNtoolkit)

import os
import numpy as np
import pandas as pd
from utils_norm.myblr import *
from statsmodels.stats import multitest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pcntoolkit.util.utils import create_design_matrix, create_bspline_basis

def nm_train(out_dir, HC_tr, HC_test, cov, idp, site, bspline_cov=None, bspline_range=None):
    """
    Train normative model using healthy control data (HC_tr) and test on held-out HC_test.
    Saves predictive distributions, performance metrics, and trained hyperparameters.
    
    Parameters:
    -----------
    out_dir : str
        Output directory to save results.
    HC_tr : pd.DataFrame
        Training data (healthy reference group).
    HC_test : pd.DataFrame
        Test data (healthy reference group).
    cov : list
        List of covariate column names.
    idp : list
        List of brain feature names to model.
    site : str
        Column name for site identifier.
    bspline_cov : list, optional
        Covariates to apply B-spline expansion.
    bspline_range : list, optional
        Ranges for each spline covariate.
    """
    remove_roi = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'Rho','pRho','pRho_fdr','MAE','RMSE','SMSE','t_stat','t_p','t_pfdr'])
    remove_roi_tvalue = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'Rho','pRho','pRho_fdr','MAE','RMSE','SMSE','t_stat','t_p','t_pfdr'])

    # Get site IDs and prepare data
    site_ids = sorted(set(HC_tr[site].to_list()))
    HC_idp_tr = HC_tr[idp]
    HC_idp_tr_np = HC_idp_tr.to_numpy()
    HC_idp_te = HC_test[idp]
    HC_idp_te_np = HC_idp_te.to_numpy()

    os.makedirs(out_dir, exist_ok=True)
    blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC','Skew','Kurtosis','Rho','pRho','MAE','RMSE','SMSE','t-stat','t_p'])

    # Process each brain feature
    for idp_num, idp_str in enumerate(idp): 
        print('Running IDP', idp_num, idp_str, ':')

        # Set up output directory for this feature
        idp_dir = os.path.join(out_dir, idp_str)
        os.makedirs(idp_dir, exist_ok=True)

        # Extract training and test data
        y_tr = HC_idp_tr_np[:,idp_num]
        y_te = HC_idp_te_np[:,idp_num]

        # Remove outliers using IQR method
        Q1 = np.quantile(y_tr, .25)
        Q3 = np.quantile(y_tr, .75)
        IQR = Q3-Q1
        lower_outliers = Q1 - 1.5 * IQR
        upper_outliers = Q3 + 1.5 * IQR
        outlier_thresh = abs(max([lower_outliers, upper_outliers], key=abs))
        nz_tr = np.abs(y_tr) < outlier_thresh
        y_tr = y_tr[nz_tr]

        # Save response variables
        resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
        resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
        np.savetxt(resp_file_tr, y_tr)
        np.savetxt(resp_file_te, y_te)

        # Create design matrices
        clinic_idx = cov
        X_tr = create_design_matrix(HC_tr[clinic_idx].loc[nz_tr], 
                                  site_ids=HC_tr[site].loc[nz_tr],
                                  basis=None, 
                                  basis_column=0)

        X_te = create_design_matrix(HC_test[clinic_idx], 
                                  site_ids=HC_test[site], 
                                  all_sites=site_ids,
                                  basis=None, 
                                  basis_column=0)

        # Add B-spline basis if specified
        if bspline_cov is not None:
            for item, (xmin, xmax) in zip(bspline_cov, bspline_range):
                B_basis = create_bspline_basis(xmin, xmax)
                X_tr = np.concatenate((X_tr, np.array([B_basis(i) for i in HC_tr[item].loc[nz_tr]])), axis=1)
                X_te = np.concatenate((X_te, np.array([B_basis(i) for i in HC_test[item]])), axis=1)

        # Save design matrices
        cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
        cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
        np.savetxt(cov_file_tr, X_tr)
        np.savetxt(cov_file_te, X_te)

        # Convert to float64 for numerical stability
        X_tr = np.array(X_tr, dtype='float64')
        y_tr = np.array(y_tr, dtype='float64')
        X_te = np.array(X_te, dtype='float64')
        y_te = np.array(y_te, dtype='float64')

        # Train model and make predictions
        hyp, Nlz = estimate(X_tr, y_tr)
        y_predicted, s2 = predict(hyp, X_tr, y_tr, X_te)
        evaluated_table, z = evaluate(hyp, y_tr, y_te, y_predicted, s2, Nlz)

        # Save model outputs
        ext = '_estimate.txt'
        np.savetxt(os.path.join(idp_dir, 'yhat' + ext), y_predicted)
        np.savetxt(os.path.join(idp_dir, 'ys2' + ext), s2)
        np.savetxt(os.path.join(idp_dir, 'Z' + ext), z)
        np.savetxt(os.path.join(idp_dir, 'hyp.txt'), hyp)

        # Calculate calibration metrics
        [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(z)

        # Store metrics
        blr_metrics.loc[len(blr_metrics)] = [idp_str, Nlz, evaluated_table['EXPV'], evaluated_table['MSLL'], 
                                           evaluated_table['BIC'], skew, kurtosis, evaluated_table['Rho'], 
                                           evaluated_table['pRho'], evaluated_table['MAE'], evaluated_table['RMSE'], 
                                           evaluated_table['SMSE'], evaluated_table['t-stat'], evaluated_table['t_p']]

    # Apply FDR corrections to p-values
    pRho_fdr = multitest.fdrcorrection(blr_metrics['pRho'], alpha=0.05, method='indep', is_sorted=False)
    blr_metrics['pRho_fdr'] = pRho_fdr[1]
    pRho_fdr = multitest.fdrcorrection(blr_metrics['t_p'], alpha=0.05, method='indep', is_sorted=False)
    blr_metrics['t_p_fdr'] = pRho_fdr[1]
    blr_metrics.to_csv(os.path.join(out_dir,'blr_metrics.csv'), index=False)
    # Exclude models not fit (based on Rho or t-stat FDR thresholds)
    for idp_num, idp_str in enumerate(idp): 
        if blr_metrics['pRho_fdr'][idp_num] > 0.05:
            remove_roi.loc[len(remove_roi)] = [blr_metrics['eid'][idp_num], blr_metrics['NLL'][idp_num], 
                                            blr_metrics['EV'][idp_num], blr_metrics['Rho'][idp_num], 
                                            blr_metrics['pRho'][idp_num], blr_metrics['pRho_fdr'][idp_num], blr_metrics['MAE'][idp_num], 
                                            blr_metrics['RMSE'][idp_num], blr_metrics['SMSE'][idp_num], 
                                            blr_metrics['t-stat'][idp_num], blr_metrics['t_p'][idp_num], 
                                            blr_metrics['t_p_fdr'][idp_num]]
        if blr_metrics['t_p_fdr'][idp_num] < 0.05:
            remove_roi_tvalue.loc[len(remove_roi_tvalue)] = [blr_metrics['eid'][idp_num], blr_metrics['NLL'][idp_num], 
                                            blr_metrics['EV'][idp_num], blr_metrics['Rho'][idp_num], 
                                            blr_metrics['pRho'][idp_num], blr_metrics['pRho_fdr'][idp_num], blr_metrics['MAE'][idp_num], 
                                            blr_metrics['RMSE'][idp_num], blr_metrics['SMSE'][idp_num], 
                                            blr_metrics['t-stat'][idp_num], blr_metrics['t_p'][idp_num],
                                            blr_metrics['t_p_fdr'][idp_num]]
    
    remove_roi.to_csv(os.path.join(out_dir,'remove_roi_rvalue.csv'), index=False)
    remove_roi_tvalue.to_csv(os.path.join(out_dir,'remove_roi_tvalue.csv'), index=False)

def test_on_sub(model_path, data_df, cov, idp, site, site_ids, patient_name, bspline_cov=None, bspline_range=None):
    """
    Makes predictions and evaluates model performance on new data.

    Parameters:
    -----------
    model_path : str
        Directory containing trained model files (hyperparameters, etc.).
    data_df : pd.DataFrame
        Data from the group to predict (HC or patients).
    cov : list
        List of covariate column names used in training.
    idp : list
        List of brain feature names to predict.
    site : str
        Column name for site identifier.
    site_ids : list
        List of all site identifiers used in training.
    patient_name : str
        Identifier for this test subject group (used in output filenames).
    bspline_cov : list, optional
        Covariates to apply B-spline expansion.
    bspline_range : list, optional
        Ranges for each spline covariate.

    Notes:
    ------
    For each brain feature:
    1. Loads trained model parameters
    2. Prepares test data with same preprocessing as training
    3. Makes predictions and evaluates performance
    4. Saves predictions and evaluation metrics
    """
    # Extract brain features
    data_idp_df = data_df[idp]
    data_idp_np = data_idp_df.to_numpy()

    # Set up output directory and metrics DataFrame
    out_dir = model_path
    blr_metrics = pd.DataFrame(columns=['eid', 'EV', 'MSLL', 'Skew', 'Kurtosis', 
                                      'Rho', 'pRho', 'MAE', 'RMSE', 'SMSE', 't-stat', 't_p'])

    # Process each brain feature
    for idp_num, idp_str in enumerate(idp): 
        print('Running IDP', idp_num, idp_str, ':')
        
        # Set up output directory for this feature
        idp_dir = os.path.join(out_dir, idp_str)
        os.makedirs(idp_dir, exist_ok=True)
            
        # Extract test data
        y_te = data_idp_np[:,idp_num]
            
        # Save response variables
        resp_file_te = os.path.join(idp_dir, f'resp_{patient_name}.txt')
        np.savetxt(resp_file_te, y_te)
                
        # Create design matrix
        cov_file_te = os.path.join(idp_dir, f'cov_bspline_{patient_name}.txt')
        clinic_idx = cov
        X_te = create_design_matrix(data_df[clinic_idx], 
                                  site_ids=data_df[site],
                                  all_sites=site_ids,
                                  basis=None, 
                                  basis_column=0)
            
        # Add B-spline basis if specified
        if bspline_cov is not None:
            for item, (xmin, xmax) in zip(bspline_cov, bspline_range):
                B_basis = create_bspline_basis(xmin, xmax)
                X_te = np.concatenate((X_te, np.array([B_basis(i) for i in data_df[item]])), axis=1)

        # Save design matrix
        np.savetxt(cov_file_te, X_te)
        X_te = np.array(X_te, dtype='float64')
        y_te = np.array(y_te, dtype='float64')

        # Load trained model parameters
        hyp = np.loadtxt(os.path.join(idp_dir, 'hyp.txt'))
        cov_file_tr = os.path.join(idp_dir, 'cov_bspline_tr.txt')
        resp_file_tr = os.path.join(idp_dir, 'resp_tr.txt')
        X_tr = np.loadtxt(cov_file_tr)
        Y_tr = np.loadtxt(resp_file_tr)

        # Make predictions
        y_predicted, s2 = predict(hyp, X_tr, Y_tr, X_te)
        evaluated_table, z = evaluate(hyp, Y_tr, y_te, y_predicted, s2)

        # Save predictions and Z-scores
        ext = f'_predict_{patient_name}.txt'
        np.savetxt(os.path.join(idp_dir, 'yhat' + ext), y_predicted)
        np.savetxt(os.path.join(idp_dir, 'ys2' + ext), s2)
        np.savetxt(os.path.join(idp_dir, 'Z' + ext), z)

        # Calculate calibration metrics
        [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(z)
        
        # Store metrics
        blr_metrics.loc[len(blr_metrics)] = [idp_str, evaluated_table['EXPV'], evaluated_table['MSLL'], 
                                           skew, kurtosis, evaluated_table['Rho'], evaluated_table['pRho'], 
                                           evaluated_table['MAE'], evaluated_table['RMSE'], evaluated_table['SMSE'], 
                                           evaluated_table['t-stat'], evaluated_table['t_p']]

    # Save metrics
    blr_metrics.to_csv(os.path.join(out_dir, f'blr_metrics_{patient_name}.csv'), index=False)

def deviation_summary(out_dir, HC, patient, HC_name, patient_name):
    """
    Compile Z-score deviations for patients and controls.
    Combines predictions and calculates deviations from normative model.
    
    Parameters:
    -----------
    out_dir : str
        Base output directory containing model results.
    HC : pd.DataFrame
        Healthy control data with brain measurements.
    patient : pd.DataFrame
        Patient data with brain measurements.
    HC_name : str
        Label for healthy controls (used in filenames).
    patient_name : str
        Label for patient group (used in filenames).
        
    Notes:
    ------
    For each brain region:
    1. Loads Z-scores from model predictions
    2. Combines HC and patient deviations
    3. Saves separate files for HC and patient deviations

    The revision analyses retain all predicted deviation scores. The model
    evaluation metrics are saved for transparency, but no ROI is removed or
    zeroed on the basis of pRho_fdr or other model-performance thresholds.
    """
    # Load model metrics
    Test_metrics = pd.read_csv(os.path.join(out_dir, 'blr_metrics.csv'))
    deviation_count = pd.DataFrame()

    # Process each brain region
    for idp_num, idp_str in enumerate(Test_metrics['eid']): 
        idp_dir = os.path.join(out_dir, idp_str)
        temp_df = pd.DataFrame(columns=[idp_str])
        
        # Load Z-scores
        z_estimate = np.loadtxt(os.path.join(idp_dir, f'Z_predict_{HC_name}.txt'))
        z_predice = np.loadtxt(os.path.join(idp_dir, f'Z_predict_{patient_name}.txt'))
        
        # Combine Z-scores
        temp_df[idp_str] = np.concatenate((z_estimate, z_predice), axis=0).squeeze()
        
        deviation_count = pd.concat([deviation_count, temp_df], axis=1)

    # Prepare patient deviations
    Proband_pat = patient['eid']
    pat_deviation = deviation_count[len(z_estimate):]
    Proband_pat.reset_index(drop=True, inplace=True)
    pat_deviation.reset_index(drop=True, inplace=True)
    pat_deviation = pd.concat([Proband_pat, pat_deviation], axis=1)

    # Prepare HC deviations
    HC_deviation = deviation_count[:len(z_estimate)]
    Proband_HC = HC['eid']
    HC_deviation.reset_index(drop=True, inplace=True)
    HC_deviation = pd.concat([Proband_HC, HC_deviation], axis=1)

    # Set poor-fitting models to None
    for idp_num, idp_str in enumerate(Test_metrics['eid']): 
        if Test_metrics.loc[idp_num,'pRho_fdr'] > 0.05:
            HC_deviation[idp_str] = None
            pat_deviation[idp_str] = None
            
    # Save results
    HC_deviation.to_csv(os.path.join(out_dir, f'{HC_name}_deviation.csv'), index=False)
    pat_deviation.to_csv(os.path.join(out_dir, f'Patient_deviation_{patient_name}.csv'), index=False)

def imputation_KNN(df, scaler=None, imputer=None):
    """
    Perform KNN imputation and standardization on data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data to impute and standardize.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-fitted scaler. If None, a new scaler is fitted.
    imputer : sklearn.impute.KNNImputer, optional
        Pre-fitted imputer. If None, a new imputer is fitted.
        
    Returns:
    --------
    tuple
        - pd.DataFrame: Imputed and standardized data
        - StandardScaler: Fitted scaler
        - KNNImputer: Fitted imputer
        
    Notes:
    ------
    Process:
    1. If no scaler provided, fits StandardScaler to data (Only for training data)
    2. Standardizes data
    3. If no imputer provided, fits KNNImputer
    4. Imputes missing values using KNN
    5. Returns processed data and fitted transformers
    """

    # Fit or use existing scaler
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(df)
    else:
        data = scaler.transform(df)
    
    # Fit or use existing imputer
    if imputer is None:
        imputer = KNNImputer(n_neighbors=10, weights='uniform')
        data = imputer.fit_transform(data)
    else:
        data = imputer.transform(data)
    
    # Convert back to DataFrame
    df_imputed = scaler.inverse_transform(data)
    
    return df_imputed, scaler, imputer

def preprocess_data(root_dir, HC_data_nm, cov, perm, split_data=True, imput_models=None):
    """
    Preprocess healthy control dataset for normative modeling.
    Handles data splitting, imputation, and feature selection.
    
    Parameters:
    -----------
    root_dir : str
        Root directory containing project data and documentation.
    HC_data_nm : pd.DataFrame
        Healthy control dataset for normative modeling.
    cov : list
        List of covariate column names.
    perm : int
        Random seed for reproducible train/test splits.
    split_data : bool, default=True
        Whether to split data into training and test sets.
    imput_models : dict, optional
        Pre-fitted imputation models. If provided, split_data must be False.
        
    Returns:
    --------
    tuple
        If split_data=True:
            - HC_train_SA, HC_test_SA: Train/test splits for surface area data
            - HC_train_CT, HC_test_CT: Train/test splits for cortical thickness data
            - HC_train_CV, HC_test_CV: Train/test splits for cortical volume data
            - idp_SA, idp_CT, idp_CV: Feature lists for each modality
            - imput_models: None
        If split_data=False:
            - NM ready exist, HC data only for testing
            - HC_SA, HC_CT, HC_CV: Processed data for each modality
            - idp_SA, idp_CT, idp_CV: Feature lists for each modality
            - imput_models: imputation models for testing data
            
    Notes:
    ------
    Process:
    1. Loads brain region mappings from documentation
    2. Optionally splits data into train/test sets
    3. Imputes missing values using KNN
    4. Returns processed data and feature lists
    
    The function handles three brain measurement modalities:
    - Surface Area (SA)
    - Cortical Thickness (CT) 
    - Cortical Volume (CV)
    """
    # Input validation
    if split_data and imput_models:
        print("Warning: split_data and imput_models cannot be true at the same time. Exiting function.")
        return
        
    # Split data if requested
    if split_data:
        HC_train, HC_test = train_test_split(HC_data_nm, test_size=0.2, 
                                           stratify=HC_data_nm[['age_group','sex']], 
                                           random_state=perm)
    else:
        HC_train = HC_data_nm
        HC_test = None

    def load_and_format_idp_map(file_path):
        """Helper function to load and format brain region mappings."""
        idp_map = pd.read_csv(file_path)
        idp_map['formatted_id'] = idp_map['idx_number'].astype(str) + '-2.0'
        return idp_map

    # Load brain region mappings
    def doc_path(filename):
        path = os.path.join(root_dir, "docs", filename)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Could not find {filename} under docs/.")

    CT_idp_map = load_and_format_idp_map(doc_path('aseg_2009_CT_formatted.csv'))
    SA_idp_map = load_and_format_idp_map(doc_path('aseg_2009_SA_formatted.csv'))
    CV_idp_map = load_and_format_idp_map(doc_path('aseg_2009_CV_formatted.csv'))

    def get_idp_list(idp_map):
        """Helper function to extract feature lists."""
        return idp_map['formatted_id'].tolist()

    # Get feature lists
    idp_SA = get_idp_list(SA_idp_map)
    idp_CT = get_idp_list(CT_idp_map)
    idp_CV = get_idp_list(CV_idp_map)

    def impute_and_select_columns(data, idp_list, cov, scaler=None, imputer=None):
        """Helper function to impute data and select relevant columns."""
        data[idp_list], scaler, imputer = imputation_KNN(data[idp_list], scaler, imputer)
        return data[['eid'] + idp_list + cov], scaler, imputer

    # Process data based on split_data flag
    if split_data:
        # Initialize imputation models dictionary
        imput_models = {'SA': {}, 'CT': {}, 'CV': {}}
        
        # Process training data
        HC_train_SA, imput_models['SA']['scaler'], imput_models['SA']['imputer'] = impute_and_select_columns(
            HC_train, idp_SA, cov)
        HC_train_CT, imput_models['CT']['scaler'], imput_models['CT']['imputer'] = impute_and_select_columns(
            HC_train, idp_CT, cov)
        HC_train_CV, imput_models['CV']['scaler'], imput_models['CV']['imputer'] = impute_and_select_columns(
            HC_train, idp_CV, cov)
        
        # Process test data using fitted models
        HC_test_SA, _, _ = impute_and_select_columns(
            HC_test, idp_SA, cov, imput_models['SA']['scaler'], imput_models['SA']['imputer'])
        HC_test_CT, _, _ = impute_and_select_columns(
            HC_test, idp_CT, cov, imput_models['CT']['scaler'], imput_models['CT']['imputer'])
        HC_test_CV, _, _ = impute_and_select_columns(
            HC_test, idp_CV, cov, imput_models['CV']['scaler'], imput_models['CV']['imputer'])
            
    else:
        # Process all data using provided models
        HC_train_SA, _, _ = impute_and_select_columns(
            HC_train, idp_SA, cov, 
            imput_models['SA']['scaler'] if imput_models else None,
            imput_models['SA']['imputer'] if imput_models else None)
        HC_train_CT, _, _ = impute_and_select_columns(
            HC_train, idp_CT, cov,
            imput_models['CT']['scaler'] if imput_models else None,
            imput_models['CT']['imputer'] if imput_models else None)
        HC_train_CV, _, _ = impute_and_select_columns(
            HC_train, idp_CV, cov,
            imput_models['CV']['scaler'] if imput_models else None,
            imput_models['CV']['imputer'] if imput_models else None)
        imput_models = None
        HC_test_SA = HC_test_CT = HC_test_CV = None

    def rename_columns(data, idp_map):
        rename_dict = dict(zip(idp_map['formatted_id'], idp_map['formatted_name']))
        return data.rename(columns=rename_dict, inplace=False)

    HC_train_SA = rename_columns(HC_train_SA, SA_idp_map)
    HC_train_CT = rename_columns(HC_train_CT, CT_idp_map)
    HC_train_CV = rename_columns(HC_train_CV, CV_idp_map)
    
    if split_data:
        HC_test_SA = rename_columns(HC_test_SA, SA_idp_map)
        HC_test_CT = rename_columns(HC_test_CT, CT_idp_map)
        HC_test_CV = rename_columns(HC_test_CV, CV_idp_map)

    idp_SA = SA_idp_map['formatted_name']
    idp_CT = CT_idp_map['formatted_name']
    idp_CV = CV_idp_map['formatted_name']

    return (HC_train_SA, HC_train_CT, HC_train_CV, 
        HC_test_SA, HC_test_CT, HC_test_CV,
        idp_SA, idp_CT, idp_CV, imput_models)

def process_and_train_model(root_dir, out_folder, cov, perm, HC_nm_data=None, HC_data=None, pat_data=None, pat_name=None, ROI_list=None, sub_name=None, split_data=True):
    """
    Full pipeline to train normative models and test on patients or new controls.
    Handles data preprocessing, model training, and prediction for multiple brain measurement modalities.
    
    Parameters:
    -----------
    root_dir : str
        Root directory of the project containing data and documentation.
    out_folder : str
        Subfolder name to store results.
    cov : list
        List of covariates to include in the model.
    perm : int
        Random seed for reproducible results.
    HC_nm_data : pd.DataFrame, optional
        Normative model reference dataset (healthy controls).
    HC_data : pd.DataFrame, optional
        Additional control data for patient comparison.
    pat_data : pd.DataFrame, optional
        Patient data to test against normative model.
    pat_name : str, optional
        Identifier for patient dataset in output files.
    ROI_list : list, optional
        Subset of regions of interest to analyze. If None, uses all regions.
    sub_name : str, optional
        Optional suffix for output filenames.
    split_data : bool, default=True
        Whether to split HC_nm_data into train/test sets.
        
    Notes:
    ------
    Process:
    1. Preprocesses data for each modality (SA, CT, CV)
    2. If split_data=True:
       - Splits HC_nm_data into train/test
       - Trains normative models on training data
       - Tests on held-out data
    3. If patient data provided:
       - Applies trained models to patient data
       - Compares with control group
       - Saves deviation scores
       
    The function handles three brain measurement modalities:
    - Surface Area (SA)
    - Cortical Thickness (CT)
    - Cortical Volume (CV)
    
    Results are saved in separate directories for each modality under out_folder.
    """
    # Remove 'site' from covariates for normative modeling
    nm_cov = [c for c in cov if c != 'site']
    modalities = ['SA', 'CT', 'CV']

    # Train models if splitting data
    if split_data:
        # Initialize dictionaries
        HC_train, HC_test, idp = {}, {}, {}
        
        # Preprocess data and get train/test splits
        HC_train['SA'], HC_train['CT'], HC_train['CV'], \
        HC_test['SA'], HC_test['CT'], HC_test['CV'], \
        idp['SA'], idp['CT'], idp['CV'], imput_models = preprocess_data(
            root_dir, HC_nm_data, cov, perm, split_data=True
        )
        
        # Override ROI list if specified
        if ROI_list is not None:
            idp['SA'] = idp['CT'] = idp['CV'] = ROI_list
            
        # Train models for each modality
        for modality in modalities:
            out_dir = os.path.join(root_dir, out_folder, f'{modality}_age_45_85', f'perm_{perm}')
            nm_train(out_dir, HC_train[modality], HC_test[modality], nm_cov, idp[modality], 'site', 
                    bspline_cov=['age'], bspline_range=[[45, 85]])

    # Process and test patient data if provided
    if HC_data is not None and pat_data is not None:
        suffix = f'_{sub_name}' if sub_name else ''

        # Preprocess control data
        HC_train, idp = {}, {}
        HC_train['SA'], HC_train['CT'], HC_train['CV'], \
        _, _, _, \
        idp['SA'], idp['CT'], idp['CV'], _ = preprocess_data(
            root_dir, HC_data, cov, perm, split_data=False, imput_models=imput_models
        )

        # Preprocess patient data
        pat_train = {}
        pat_train['SA'], pat_train['CT'], pat_train['CV'], \
        _, _, _, _, _, _, _ = preprocess_data(
            root_dir, pat_data, cov, perm, split_data=False, imput_models=imput_models
        )
        
        # Override ROI list if specified
        if ROI_list is not None:
            idp['SA'] = idp['CT'] = idp['CV'] = ROI_list
            
        # Get site IDs for model application
        site_ids = sorted(set(HC_nm_data['site']) if HC_nm_data is not None else set(HC_data['site']))

        # Apply models to each modality
        for modality in modalities:
            out_dir = os.path.join(root_dir, out_folder, f'{modality}_age_45_85', f'perm_{perm}')

            # Test on patient data
            test_on_sub(out_dir, pat_train[modality], nm_cov, idp[modality], 'site', site_ids,
                       f'{pat_name}{suffix}', bspline_cov=['age'], bspline_range=[[45, 85]])

            # Test on control data
            test_on_sub(out_dir, HC_train[modality], nm_cov, idp[modality], 'site', site_ids,
                       f'HC{suffix}', bspline_cov=['age'], bspline_range=[[45, 85]])
            
            # Calculate deviations
            deviation_summary(out_dir, HC_train[modality], pat_train[modality], 'HC', pat_name)
