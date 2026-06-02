"""
Bayesian linear regression (BLR) with warping for normative modeling.
Adapted from PCNtoolkit (https://github.com/amarquand/PCNtoolkit)

This module implements a Bayesian Linear Regression model with non-linear warping
to handle non-Gaussian distributions. The model is particularly suited for normative
modeling of brain measurements.

Key features:
- Non-linear warping using sinh/arcsinh transformations
- Automatic relevance determination for feature selection
- Comprehensive model evaluation metrics

Example:
    >>> # Train model
    >>> hyp, nlZ = estimate(X_train, y_train)
    >>> # Make predictions
    >>> y_pred, s2 = predict(hyp, X_train, y_train, X_test)
    >>> # Evaluate performance
    >>> metrics, z_scores = evaluate(hyp, y_train, y_test, y_pred, s2)
"""

import numpy as np
from scipy import stats
from scipy import optimize, linalg
from scipy.linalg import LinAlgError
from scipy.stats import norm, ttest_rel

def compute_pearsonr(A, B):
    """
    Compute column-wise Pearson correlation between A and B.

    Parameters:
    -----------
    A, B : np.ndarray
        2D arrays of the same shape (n_samples, n_features)
        Must be numeric arrays with no missing values

    Returns:
    --------
    Rho : np.ndarray
        Pearson correlation coefficients for each feature
    pRho : np.ndarray
        Two-tailed p-values for each correlation

    Notes:
    ------
    Uses Fisher's r-to-z transformation for comparison.
    """
    # Input validation
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if A.shape != B.shape:
        raise ValueError("Input arrays must have the same shape")
    if np.isnan(A).any() or np.isnan(B).any():
        raise ValueError("Input arrays must not contain NaN values")

    N = A.shape[0]

    # First mean centre
    Am = A - np.mean(A, axis=0)
    Bm = B - np.mean(B, axis=0)
    # Then normalize
    An = Am / np.sqrt(np.sum(Am**2, axis=0))
    Bn = Bm / np.sqrt(np.sum(Bm**2, axis=0))
    del(Am, Bm)

    Rho = np.sum(An * Bn, axis=0)
    del(An, Bn)

    # Fisher r-to-z transformation
    Zr = (np.arctanh(Rho) - np.arctanh(0)) * np.sqrt(N - 3)
    N = stats.norm()
    pRho = 2*N.cdf(-np.abs(Zr))

    return Rho, pRho

def warp_f(y, gamma):
    """
    Warp function using sinh/arcsinh transformation.

    Parameters:
    -----------
    y : np.ndarray
        Input values to be warped
    gamma : np.ndarray
        Warping parameters [epsilon, log(b)]
        epsilon controls asymmetry
        b controls the strength of warping

    Returns:
    --------
    np.ndarray
        Warped values

    Notes:
    ------
    The warping function is: sinh(b * arcsinh(y) - a)
    where a = -epsilon * b
    This transformation can handle both symmetric and asymmetric non-Gaussianity.
    """
    # Input validation
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if not isinstance(gamma, np.ndarray) or len(gamma) != 2:
        raise ValueError("gamma must be a numpy array of length 2")

    epsilon = gamma[0]
    b = np.exp(gamma[1])
    a = -epsilon*b

    y = np.sinh(b * np.arcsinh(y) - a)
    return y

def warp_df(y_unwarped, gamma):
    """
    Derivative of the warp function.
    """
    epsilon = gamma[0]
    b = np.exp(gamma[1])
    a = -epsilon*b

    dx = (b *np.cosh(b * np.arcsinh(y_unwarped) - a))/np.sqrt(1 + y_unwarped ** 2)

    return dx

def warp_invf(y, gamma):
    """
    Inverse warp function (for prediction and visualization).
    """
    epsilon = gamma[0]
    b = np.exp(gamma[1])
    a = -epsilon*b
    
    x = np.sinh((np.arcsinh(y)+a)/b)
    
    return x


def warp_predictions(mu, s2, gamma, percentiles=[0.025, 0.975]):
    """
    Compute warped predictions with confidence intervals.
    """
    N = norm
    Z = N.ppf(percentiles)

    median = warp_invf(mu, gamma)

    # compute the predictive intervals (non-stationary)
    pred_interval = np.zeros((len(mu), len(Z)))
    for i, z in enumerate(Z):
        pred_interval[:,i] = warp_invf(mu + np.sqrt(s2)*z, gamma)

    return median, pred_interval


def loglik(hyp, x_tr, y_tr, Xv):
    """
    Compute negative log-likelihood for warped BLR model.
    """
    N = x_tr.shape[0]
    D = x_tr.shape[1]

    beta = np.asarray([np.exp(hyp[0])]) 
    gamma = hyp[1:3]
    alpha = np.exp(hyp[3:])

    delta = np.exp(gamma[1])
    beta = beta/(delta**2)

    y_unwarped = y_tr
    y = warp_f(y_tr, gamma)

    lambda_n_vec = np.ones(N)*beta
    Sigma_a = np.diag(np.ones(D)) / alpha
    Lambda_a = np.diag(np.ones(D)) * alpha
    try: 
        XtLambda_n = x_tr.T * lambda_n_vec
        A = XtLambda_n.dot(x_tr) + Lambda_a
        invAXt = linalg.solve(A, x_tr.T, check_finite=False)
        m = (invAXt * lambda_n_vec).dot(y)

    except ValueError:
        nlZ = 1/np.finfo(float).eps
        return nlZ

    logdetSigma_n = sum(np.log(1 / lambda_n_vec))
    logdetSigma_a = sum(np.log(np.diag(Sigma_a)))
    X_y_t_sLambda_n = (y-x_tr.dot(m)) * np.sqrt(lambda_n_vec)
    try:
        logdetA = 2*sum(np.log(np.diag(np.linalg.cholesky(A))))
    except (ValueError, LinAlgError):
        nlZ = 1/np.finfo(float).eps
        return nlZ

    nlZ = -0.5 * (-N*np.log(2*np.pi) -
                logdetSigma_n -
                logdetSigma_a -
                X_y_t_sLambda_n.T.dot(X_y_t_sLambda_n) -
                m.T.dot(Lambda_a).dot(m) -
                logdetA
                )

    nlZ = nlZ - sum(np.log(warp_df(y_unwarped, gamma)))

    if not np.isfinite(nlZ):
        nlZ = 1/np.finfo(float).eps

    return nlZ

def penalized_loglik(hyp, x_tr, y_tr, Xv, l):
    """
    Penalized likelihood with L2 regularization for optimization.
    """
    L = loglik(hyp, x_tr, y_tr, Xv) + l * np.sqrt(sum(hyp**2))
    return L

def estimate(x_tr, y_tr):
    """
    Estimates model hyperparameters using L-BFGS-B optimization.

    Parameters:
    -----------
    x_tr : np.ndarray
        Training features (n_samples, n_features)
    y_tr : np.ndarray
        Training targets (n_samples,)

    Returns:
    --------
    hyp : np.ndarray
        Optimized hyperparameters [log(beta), gamma_1, gamma_2, log(alpha)]
        beta: noise precision
        gamma: warping parameters
        alpha: feature precision
    nlZ : float
        Negative log-likelihood at optimum

    Notes:
    ------
    Uses L-BFGS-B optimization with L2 regularization.
    Handles numerical instabilities through error catching.
    """
    # Input validation
    if not isinstance(x_tr, np.ndarray) or not isinstance(y_tr, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if x_tr.shape[0] != y_tr.shape[0]:
        raise ValueError("Number of samples must match between x_tr and y_tr")
    if np.isnan(x_tr).any() or np.isnan(y_tr).any():
        raise ValueError("Input arrays must not contain NaN values")

    # alpha, beta, gamma_1, gamma_2 (gamma are for warped y)
    _n_params = 1 + 1 + 2

    epsilon = 0.1
    l = 0.1
    Xv = None

    hyp0 = np.zeros(_n_params)
    all_hyp_i = [hyp0]
    def store(X):
        hyp = X
        all_hyp_i.append(hyp)
    try:
        out = optimize.fmin_l_bfgs_b(penalized_loglik, x0=hyp0,
                                    args=(x_tr, y_tr, Xv, l), approx_grad=True,
                                    epsilon=epsilon, callback=store)
    except np.linalg.LinAlgError:
        out = optimize.fmin_l_bfgs_b(penalized_loglik, x0=all_hyp_i[-1],
                                    args=(x_tr, y_tr, Xv, l), approx_grad=True,
                                    epsilon=epsilon)
    
    hyp = out[0]
    nlZ = out[1]

    return hyp, nlZ

def predict(hyp, x_tr, y_tr, x_test):
    """
    Predicts warped BLR output on test set using trained model.

    Parameters:
    -----------
    hyp : np.ndarray
        Model hyperparameters [log(beta), gamma_1, gamma_2, log(alpha)]
    x_tr : np.ndarray
        Training features (n_samples, n_features)
    y_tr : np.ndarray
        Training targets (n_samples,)
    x_test : np.ndarray
        Test features (n_test_samples, n_features)

    Returns:
    --------
    ys : np.ndarray
        Predicted mean values
    s2 : np.ndarray
        Predictive variances

    Notes:
    ------
    The prediction includes both the mean prediction and uncertainty estimates.
    The variance accounts for both model uncertainty and noise.
    """
    # Input validation
    if not isinstance(hyp, np.ndarray) or len(hyp) != 4:
        raise ValueError("hyp must be a numpy array of length 4")
    if not isinstance(x_tr, np.ndarray) or not isinstance(y_tr, np.ndarray):
        raise TypeError("Training inputs must be numpy arrays")
    if not isinstance(x_test, np.ndarray):
        raise TypeError("Test inputs must be numpy arrays")
    if x_tr.shape[0] != y_tr.shape[0]:
        raise ValueError("Number of samples must match between x_tr and y_tr")
    if x_tr.shape[1] != x_test.shape[1]:
        raise ValueError("Number of features must match between training and test data")
    if np.isnan(x_tr).any() or np.isnan(y_tr).any() or np.isnan(x_test).any():
        raise ValueError("Input arrays must not contain NaN values")

    N = x_tr.shape[0]
    D = x_tr.shape[1]

    Xv = None
    l = 0.1

    beta = np.asarray([np.exp(hyp[0])]) 
    gamma = hyp[1:3]
    alpha = np.exp(hyp[3:])

    delta = np.exp(gamma[1])
    beta = beta/(delta**2)

    y_warped_tr = warp_f(y_tr, gamma)

    lambda_n_vec = np.ones(N)*beta
    Sigma_a = np.diag(np.ones(D)) / alpha
    Lambda_a = np.diag(np.ones(D)) * alpha

    XtLambda_n = x_tr.T * lambda_n_vec
    A = XtLambda_n.dot(x_tr) + Lambda_a
    invAXt = linalg.solve(A, x_tr.T, check_finite=False)
    m = (invAXt * lambda_n_vec).dot(y_warped_tr)

    N_test = x_test.shape[0]
    
    ys = x_test.dot(m)
    s2n = 1/beta
    
    s2 = s2n + np.sum(x_test*linalg.solve(A, x_test.T).T, axis=1)

    return ys, s2

def export_m(hyp, x_tr, y_tr):
    """
    Exports mean vector m (posterior weights) of the trained model.
    """
    N = x_tr.shape[0]
    D = x_tr.shape[1]

    beta = np.asarray([np.exp(hyp[0])]) 
    gamma = hyp[1:3]
    alpha = np.exp(hyp[3:])

    delta = np.exp(gamma[1])
    beta = beta/(delta**2)

    y_warped_tr = warp_f(y_tr, gamma)

    lambda_n_vec = np.ones(N)*beta
    Sigma_a = np.diag(np.ones(D)) / alpha
    Lambda_a = np.diag(np.ones(D)) * alpha

    XtLambda_n = x_tr.T * lambda_n_vec
    A = XtLambda_n.dot(x_tr) + Lambda_a
    invAXt = linalg.solve(A, x_tr.T, check_finite=False)
    m = (invAXt * lambda_n_vec).dot(y_warped_tr)
    return m

def evaluate(hyp, y_tr, y_te, y_predicted, s2, Nlz=None):
    """
    Evaluate model performance and calibration using test set.

    Parameters:
    -----------
    hyp : np.ndarray
        Model hyperparameters
    y_tr : np.ndarray
        Training targets
    y_te : np.ndarray
        Test targets
    y_predicted : np.ndarray
        Model predictions on test set
    s2 : np.ndarray
        Predictive variances
    Nlz : float, optional
        Negative log-likelihood from training

    Returns:
    --------
    result_evaluate : dict
        Dictionary containing performance metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Square Error
        - Rho: Pearson correlation
        - pRho: p-value for correlation
        - SMSE: Standardized Mean Square Error
        - EXPV: Explained Variance
        - MSLL: Mean Standardized Log Loss
        - BIC: Bayesian Information Criterion (if Nlz provided)
        - t-stat: t-statistic for paired t-test
        - t_p: p-value for t-test
    Z : np.ndarray
        Normalized residuals (Z-scores)

    Notes:
    ------
    The evaluation includes both point prediction accuracy and
    uncertainty calibration metrics.
    """
    # Input validation
    if not isinstance(hyp, np.ndarray):
        raise TypeError("hyp must be a numpy array")
    if not all(isinstance(x, np.ndarray) for x in [y_tr, y_te, y_predicted, s2]):
        raise TypeError("All inputs must be numpy arrays")
    if not all(x.shape[0] == y_te.shape[0] for x in [y_predicted, s2]):
        raise ValueError("Test data dimensions must match")
    if Nlz is not None and not isinstance(Nlz, (int, float)):
        raise TypeError("Nlz must be a scalar")

    result_evaluate = dict()
    gamma = hyp[1:3]

    y_warped_tr = warp_f(y_tr, gamma)
    y_warped_te = warp_f(y_te, gamma)

    feature_num = 1
    MAE = np.mean(abs(y_warped_te - y_predicted), axis=0)
    result_evaluate['MAE'] = MAE
    MSE = np.mean((y_warped_te - y_predicted)**2, axis=0)
    result_evaluate['RMSE'] = np.sqrt(MSE)

    Rho = np.zeros(feature_num)
    pRho = np.ones(feature_num)    
    Rho, pRho = compute_pearsonr(y_warped_te, y_predicted)
    result_evaluate['Rho'] = Rho
    result_evaluate['pRho'] = pRho

    result_evaluate['SMSE'] = MSE / np.var(y_warped_te, axis=0)

    result_evaluate['EXPV'] = 1 - (y_warped_te - y_predicted).var(axis = 0) / y_warped_te.var(axis = 0)
    Y_train_mean = np.repeat(np.mean(y_warped_tr), y_warped_te.shape[0], axis = 0)
    Y_train_sig = np.repeat(np.std(y_warped_tr)**2, y_warped_te.shape[0], axis = 0)

    result_evaluate['MSLL'] = np.mean(0.5 * np.log(2 * np.pi * s2) + (y_warped_te - y_predicted)**2 / (2 * s2) - 
                       0.5 * np.log(2 * np.pi * Y_train_sig) - (y_warped_te - Y_train_mean)**2 / (2 * Y_train_sig), axis = 0)

    test_perms = ttest_rel(y_warped_te, y_predicted)
    result_evaluate['t-stat'] = test_perms.statistic
    result_evaluate['t_p'] = test_perms.pvalue

    n = y_tr.shape[0]
    k = len(hyp)
    if Nlz is not None:
        result_evaluate['BIC'] = k * np.log(n) + 2 * Nlz
        result_evaluate['NLL'] = Nlz
    
    Z = (y_warped_te - y_predicted) / np.sqrt(s2) 

    return result_evaluate, Z

def calibration_descriptives(x):
    """
    Compute skew, kurtosis, and their standard errors for residual Z-scores.

    Parameters:
    -----------
    x : np.ndarray
        Array of Z-scores to analyze

    Returns:
    --------
    cd : list
        List containing:
        - skew: Sample skewness
        - sdskew: Standard error of skewness
        - kurtosis: Sample kurtosis
        - sdkurtosis: Standard error of kurtosis
        - semean: Standard error of mean
        - sesd: Standard error of standard deviation

    Notes:
    ------
    These statistics help assess if the residuals follow a standard normal
    distribution, which is expected for well-calibrated uncertainty estimates.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")
    if np.isnan(x).any():
        raise ValueError("x must not contain NaN values")

    n = np.shape(x)[0]
    m1 = np.mean(x)
    m2 = sum((x-m1)**2)
    m3 = sum((x-m1)**3)
    m4 = sum((x-m1)**4)
    s1 = np.std(x)
    skew = n * m3 /( n-1 )/( n-2 )/ s1**3
    sdskew = np.sqrt( 6*n*(n-1) / ((n-2)*(n+1)*(n+3)) )
    kurtosis = (n*(n+1)*m4 - 3*m2**2*(n-1)) / ((n-1)*(n-2)*(n-3)*s1**4)
    sdkurtosis = np.sqrt( 4*(n**2-1) * sdskew**2 / ((n-3)*(n+5)) )
    semean = np.sqrt(np.var(x)/n)
    sesd = s1/np.sqrt(2*(n-1))
    cd = [skew, sdskew, kurtosis, sdkurtosis, semean, sesd]
    return cd