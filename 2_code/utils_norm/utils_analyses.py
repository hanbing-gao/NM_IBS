import os
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from scipy.stats import t, ttest_ind, levene, norm, shapiro
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests

def prepare_data_for_long_structure(HC_deviation, Pat_deviation, idp):
    """
    Prepare data for linear mixed effects model analysis by combining HC and Patient data.
    
    Parameters:
    -----------
    HC_deviation : pd.DataFrame
        Healthy control deviation data
    Pat_deviation : pd.DataFrame
        Patient deviation data
    idp : list
        List of brain area identifiers to analyze
        
    Returns:
    --------
    pd.DataFrame
        Long format data ready for mixed effects analysis, containing:
        - Brain measurements for each region
        - Group identifiers
        - Sex information
        - Hemisphere indicators
    """
    # Select relevant columns
    HC_deviation = HC_deviation[['eid', 'Group', 'sex'] + idp]
    Pat_deviation = Pat_deviation[['eid', 'Group', 'sex'] + idp]
    
    # Combine all data
    data = pd.concat([HC_deviation, Pat_deviation], axis=0, ignore_index=True)
    
    # Convert categorical variables
    data['sex'] = data['sex'].astype(int).astype('category')
    data['Group'] = data['Group'].astype(int).astype('category')

    # Get all brain areas without hemisphere prefix
    brain_areas = set()
    for col in idp:
        if col.startswith('rh_') or col.startswith('lh_'):
            brain_areas.add(col[3:])

    # Check which brain areas have both hemispheres
    complete_areas = set()
    for area in brain_areas:
        if f'rh_{area}' in idp and f'lh_{area}' in idp:
            complete_areas.add(area)

    # Create list of columns to keep
    cols_to_keep = ['eid', 'Group', 'sex']
    for area in complete_areas:
        cols_to_keep.append(f'rh_{area}')
        cols_to_keep.append(f'lh_{area}')

    # Filter data to only include brain areas with both hemispheres
    data = data[cols_to_keep]
    idp = [col for col in idp if col[3:] in complete_areas]

    # Convert from wide to long format
    data_long = pd.melt(
        data,
        id_vars=['eid', 'Group', 'sex'],  # Columns to keep
        var_name='BrainArea',                  # Name for the melted variable column
        value_name='Thickness'                 # Name for the melted value column
    )

    data_long['Hemisphere'] = data_long['BrainArea'].apply(
        lambda x: 'Left' if 'lh_' in x else 'Right'
    )

    data_long['BrainArea'] = data_long['BrainArea'].apply(
        lambda x: ''+x[3:]
    )

    data_long['Hemisphere'] = data_long['Hemisphere'].astype('category')

    data_long = data_long.reset_index(drop=True)
    
    return data_long

def extract_effect(result_table, label, roi):
    """
    Extract effect size and statistics from result table.
    
    Parameters:
    -----------
    result_table : pd.DataFrame
        Results table from statistical analysis
    label : str
        Label for the effect to extract
    roi : str
        Region of interest identifier
        
    Returns:
    --------
    dict
        Dictionary containing effect statistics
    """
    return {'roi': roi, **{k: float(v) for k, v in result_table.loc[label].items()}}

def run_mixed_effects_analysis(data_long, results_dir):
    """
    Run mixed effects analysis on long format data to examine group, sex, and hemisphere effects.
    
    Parameters:
    -----------
    data_long : pd.DataFrame
        Long format data containing:
        - Brain measurements (value)
        - Group identifiers
        - Sex information
        - Hemisphere indicators
        - ROI (brain region) labels
    results_dir : str
        Directory to save analysis results
        
    Notes:
    ------
    Analyzes:
    - Main effects: Group, Sex, Hemisphere
    - Two-way interactions: Group×Sex, Group×Hemisphere, Sex×Hemisphere
    - Three-way interaction: Group×Sex×Hemisphere
    
    Results are saved as CSV files with FDR-corrected p-values.
    """
    # Initialize dictionary to store results
    result_dict = {
        'main_effect_group_1': [], 'main_effect_group_2': [], 'main_effect_group_3': [],
        'main_effect_sex': [], 'main_effect_hemisphere': [], 'interaction_sex_hemisphere': [],
        'interaction_group_sex_1': [], 'interaction_group_sex_2': [], 'interaction_group_sex_3': [],
        'interaction_group_hemisphere_1': [], 'interaction_group_hemisphere_2': [], 'interaction_group_hemisphere_3': [],
        'interaction_group_sex_hemisphere_1': [], 'interaction_group_sex_hemisphere_2': [], 'interaction_group_sex_hemisphere_3': []
    }

    for area in data_long['BrainArea'].unique():
        subset = data_long[data_long['BrainArea'] == area].copy()
        subset.dropna(inplace=True)
        if subset['Hemisphere'].nunique() == 2:
            formula = "Thickness ~ Group * Hemisphere * sex" 
            model = smf.mixedlm(formula, subset, groups=subset["eid"])
            result = model.fit()
            table = result.summary().tables[1]
            result_dict['main_effect_sex'].append(extract_effect(table, 'sex[T.1]', area))
            result_dict['main_effect_hemisphere'].append(extract_effect(table, 'Hemisphere[T.Right]', area))
            result_dict['main_effect_group_1'].append(extract_effect(table, 'Group[T.1]', area))
            result_dict['main_effect_group_2'].append(extract_effect(table, 'Group[T.2]', area))
            result_dict['main_effect_group_3'].append(extract_effect(table, 'Group[T.3]', area))

            # Two-way interactions
            result_dict['interaction_group_sex_1'].append(extract_effect(table, 'Group[T.1]:sex[T.1]', area))
            result_dict['interaction_group_sex_2'].append(extract_effect(table, 'Group[T.2]:sex[T.1]', area))
            result_dict['interaction_group_sex_3'].append(extract_effect(table, 'Group[T.3]:sex[T.1]', area))
            result_dict['interaction_sex_hemisphere'].append(extract_effect(table, 'Hemisphere[T.Right]:sex[T.1]', area))
            result_dict['interaction_group_hemisphere_1'].append(extract_effect(table, 'Group[T.1]:Hemisphere[T.Right]', area))
            result_dict['interaction_group_hemisphere_2'].append(extract_effect(table, 'Group[T.2]:Hemisphere[T.Right]', area))
            result_dict['interaction_group_hemisphere_3'].append(extract_effect(table, 'Group[T.3]:Hemisphere[T.Right]', area))
            
            # Three-way interactions
            result_dict['interaction_group_sex_hemisphere_1'].append(extract_effect(table, 'Group[T.1]:Hemisphere[T.Right]:sex[T.1]', area))
            result_dict['interaction_group_sex_hemisphere_2'].append(extract_effect(table, 'Group[T.2]:Hemisphere[T.Right]:sex[T.1]', area))
            result_dict['interaction_group_sex_hemisphere_3'].append(extract_effect(table, 'Group[T.3]:Hemisphere[T.Right]:sex[T.1]', area))

    # Convert results to DataFrames
    for key in result_dict:
        result_dict[key] = pd.DataFrame(result_dict[key])

    # Apply FDR correction to p-values
    for key in result_dict:
        if not result_dict[key].empty and 'P>|z|' in result_dict[key].columns:
            result_dict[key]['FDR_P'] = multipletests(result_dict[key]['P>|z|'], alpha=0.05, method='fdr_bh')[1]

    # Save results to CSV files
    for key, df in result_dict.items():
        df.to_csv(os.path.join(results_dir, f'{key}.csv'), index=False)

    return

def t_test_levene(data1, data2, alternative='two-sided'):
    """
    Perform t-test with Levene's test for equal variances.
    
    Parameters:
    -----------
    data1, data2 : array-like
        Arrays of data to compare
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
        
    Returns:
    --------
    tuple
        (t-statistic, p-value, Levene's test statistic, Levene's p-value)
    """
    # Remove NaN values
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    # Perform Levene's test
    levene_stat, levene_p = levene(data1, data2)
    
    # Perform t-test
    t_stat, p_val = ttest_ind(data1, data2, equal_var=(levene_p > 0.05), alternative=alternative)
    
    return t_stat, p_val, levene_stat, levene_p

def effect_size_cohens_d(group1, group2, alpha=0.05):
    """
    Calculate Cohen's d effect size with confidence intervals.
    
    Parameters:
    -----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    alpha : float, optional
        Significance level for confidence intervals
        
    Returns:
    --------
    tuple
        (Cohen's d, lower CI, upper CI)
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    cohen_d = (mean1 - mean2) / pooled_std
    dof = n1 + n2 - 2

    # Calculate confidence intervals
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (cohen_d**2 / (2 * (n1 + n2))))
    ci_low = cohen_d - t.ppf(1 - alpha / 2, dof) * se_d
    ci_high = cohen_d + t.ppf(1 - alpha / 2, dof ) * se_d

    return cohen_d, ci_low, ci_high

def hedges_g(group1, group2, alpha=0.05):
    """
    Calculate Hedges' g effect size with bias correction.
    
    Parameters:
    -----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    alpha : float, optional
        Significance level for confidence intervals
        
    Returns:
    --------
    tuple
        (Cohen's d, Hedges' g, lower CI, upper CI)
    """
    # Calculate Cohen's d first
    d = effect_size_cohens_d(group1, group2)
    
    # Calculate Hedges' g with bias correction
    n1, n2 = len(group1), len(group2)
    dof = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * dof - 1))
    g = d * correction_factor
    
    # Calculate confidence intervals
    se_g = np.sqrt((n1 + n2) / (n1 * n2) + (g**2 / (2 * dof)))
    ci_low = g - t.ppf(1 - alpha / 2, dof) * se_g
    ci_high = g + t.ppf(1 - alpha / 2, dof) * se_g
    
    return d, g, ci_low, ci_high

def calculate_power(sample_size_per_group, alpha=0.05, power=0.8):
    """
    Calculate minimum detectable effect size for given sample size and power.
    
    Parameters:
    -----------
    sample_size_per_group : int
        Sample size per group
    alpha : float, optional
        Significance level
    power : float, optional
        Desired power
        
    Returns:
    --------
    float
        Minimum detectable effect size
    """
    analysis = TTestIndPower()
    min_detectable_effect_size = analysis.solve_power(nobs1=sample_size_per_group, alpha=alpha, power=power, alternative='two-sided')
    return min_detectable_effect_size

def equivalence_test(group1, group2, margin=0.2):
    """
    Perform equivalence test using TOST (Two One-Sided Tests).
    
    Parameters:
    -----------
    group1, group2 : array-like
        Arrays of data to compare
    margin : float
        Equivalence margin (default: 0.2)
        
    Returns:
    --------
    tuple
        (effect size, CI lower bound, CI upper bound, TOST results)
    """
    pooled_std = np.sqrt(((len(group1) - 1) * np.std(group1, ddof=1)**2 + (len(group2) - 1) * np.std(group2, ddof=1)**2) / (len(group1) + len(group2) - 2))
    tost_results_margin = sms.ttost_ind(group1, group2, -margin * pooled_std, margin * pooled_std)

    d, ci_low, ci_high = effect_size_cohens_d(group1, group2)

    return d, ci_low, ci_high, tost_results_margin

# Correlation analysis
def fisher_z(r):
    """
    Convert correlation coefficient to Fisher's Z-score.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
        
    Returns:
    --------
    float
        Fisher's Z-score
    """
    return 0.5 * np.log((1 + r) / (1 - r))

def calculate_correlation_and_z_scores(merged_data, roi_list, demo_columns):
    """
    Calculate correlations and z-scores between ROIs and demographic variables.
    
    Parameters:
    -----------
    merged_data : pd.DataFrame
        Combined data containing ROIs and demographic variables
    roi_list : pd.DataFrame
        List of ROIs to analyze
    demo_columns : list
        List of demographic column names
        
    Returns:
    --------
    dict
        Dictionary of results by demographic variable
    """
    # Create a dictionary to store DataFrames for each demo column
    results_by_demo = {}
    merged_data = merged_data.copy()
    # Filter out rows with missing data in the specified columns
    merged_data.dropna(how='all', axis=1, inplace=True)
    merged_data.dropna(inplace=True)
    
    # Create a dictionary for each demo column
    for demo_col in demo_columns:
        # Initialize empty dictionary for this demo column
        roi_results = {}
        
        for roi in roi_list['ROI']:
            if roi in merged_data.columns:
                # Check normality of both variables using Shapiro-Wilk test
                _, p_val_demo = shapiro(merged_data[demo_col])
                _, p_val_roi = shapiro(merged_data[roi])
                
                # If both variables are normally distributed (p > 0.05), use Pearson
                # Otherwise use Spearman
                if p_val_demo > 0.05 and p_val_roi > 0.05:
                    corr = merged_data[demo_col].corr(merged_data[roi], method='pearson')
                    corr_method = 'pearson'
                else:
                    corr = merged_data[demo_col].corr(merged_data[roi], method='spearman') 
                    corr_method = 'spearman'
                
                n = len(merged_data)
                
                # Calculate z-score
                z = fisher_z(corr)
                
                # Calculate p-value
                p_val = 2 * (1 - norm.cdf(np.abs(corr * np.sqrt((n - 2) / (1 - corr**2)))))
                
                # Store results for this ROI
                roi_results[roi] = {
                    'corr_method':corr_method,
                    'r': corr,
                    'z_score': z,
                    'p_value': p_val,
                    'sample_size': n
                }
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame.from_dict(roi_results, orient='index')
        results_by_demo[demo_col] = df
    
    return results_by_demo

def calculate_and_compare_correlations(results1, results2, demo_col, label1='Group1', label2='Group2'):
    """
    Calculate and compare correlations between two groups.
    
    Parameters:
    -----------
    results1 : dict
        Results from first group
    results2 : dict
        Results from second group
    demo_col : str
        Demographic column name
    label1 : str, optional
        Label for first group
    label2 : str, optional
        Label for second group
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    # Extract the demographic-specific data
    data1 = results1[demo_col]
    data2 = results2[demo_col]

    # Compute z-diff and p-values
    z_diff = data1['z_score'] - data2['z_score']
    z_diff_se = np.sqrt(1 / (data1['sample_size'] - 3) + 1 / (data2['sample_size'] - 3))
    z_test = z_diff / z_diff_se
    diff_p_values = 2 * (1 - norm.cdf(np.abs(z_test)))

    # Construct results table
    results = pd.DataFrame(index=data1.index)

    results[f'{label1}_method'] = data1['corr_method']
    results[f'{label1}_r'] = data1['r']
    results[f'{label1}_z'] = data1['z_score']
    results[f'{label1}_p'] = data1['p_value']
    results[f'{label1}_p_fdr'] = multipletests(data1['p_value'], method='fdr_bh')[1]
    results[f'{label1}_n'] = data1['sample_size']

    results[f'{label2}_method'] = data2['corr_method']
    results[f'{label2}_r'] = data2['r']
    results[f'{label2}_z'] = data2['z_score']
    results[f'{label2}_p'] = data2['p_value']
    results[f'{label2}_p_fdr'] = multipletests(data2['p_value'], method='fdr_bh')[1]
    results[f'{label2}_n'] = data2['sample_size']

    results['z_diff'] = z_diff
    results['diff_p'] = diff_p_values
    results['diff_p_fdr'] = multipletests(diff_p_values, method='fdr_bh')[1]

    return results