#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:48:40 2025

@author: lena
"""

import numpy as np
import pandas as pd
from scipy.stats import beta, kendalltau, spearmanr
from scipy.optimize import minimize
from copulas.bivariate import Clayton, Gumbel, Frank
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Data Loading and Cleaning
def load_and_clean_data(file_path):
    """
    Load and clean the data to ensure validity:
    - x > 0, y > 0
    - tau1 > x, tau2 > y
    """
    data = pd.read_excel(file_path)
    data_points = data[['x', 'y', 'Tau 1', 'Tau 2']].dropna().values
    cleaned_data = data_points[
        (data_points[:, 0] > 0) & (data_points[:, 1] > 0) &
        (data_points[:, 2] > data_points[:, 0]) & (data_points[:, 3] > data_points[:, 1])
    ]
    return cleaned_data

# Step 2: Fit Beta Distribution for Marginals
def fit_beta_distribution(data):
    """
    Fit Beta distribution to the given data (x or y).
    Returns the fitted alpha and beta parameters.
    """
    def neg_log_likelihood_beta(params, data):
        alpha, beta_param = params
        if alpha <= 0 or beta_param <= 0 or np.any(data <= 0) or np.any(data >= 1):
            return np.inf
        log_likelihood = np.sum(beta.logpdf(data, alpha, beta_param))
        return -log_likelihood

    result = minimize(
        neg_log_likelihood_beta,
        [1.0, 1.0],
        args=(data,),
        method='L-BFGS-B',
        bounds=[(1e-3, None), (1e-3, None)]
    )
    if result.success:
        return result.x
    else:
        raise ValueError(f"Beta distribution fitting failed: {result.message}")

# Step 3: Interaction Term Application
def apply_interaction_term(x_simulated, y_simulated, tau1, tau2, gamma):
    """
    Apply the interaction term to the simulated data from the Copula.
    """
    interaction_weights = np.exp(-0.5 * gamma * (tau2 * x_simulated - tau1 * y_simulated) ** 2)
    x_interaction = x_simulated * interaction_weights
    y_interaction = y_simulated * interaction_weights
    return x_interaction, y_interaction

# Step 4: Copula Log-Likelihood Calculation
def calculate_log_likelihood_with_interaction(copula, u, v, x, y, tau1, tau2, gamma):
    """
    Calculate the log-likelihood of a fitted Copula model with an interaction term.
    """
    copula_density = copula.probability_density(np.column_stack([u, v]))
    interaction_term = np.exp(-0.5 * gamma * (tau2 * x - tau1 * y) ** 2)
    log_likelihood = np.sum(np.log(copula_density * interaction_term + 1e-10))  # Avoid log(0)
    return log_likelihood

# Step 5: AIC Calculation
def calculate_aic(log_likelihood, num_params):
    """
    Calculate the AIC for the model.
    """
    return 2 * num_params - 2 * log_likelihood

# Step 6: Copula Fitting and AIC Evaluation
def fit_copulas_and_calculate_aic(u, v, x, y, tau1, tau2, gamma):
    """
    Fit multiple Copulas and calculate their AIC values.
    """
    copulas = {
        "Clayton": Clayton(),
        "Gumbel": Gumbel(),
        "Frank": Frank()
    }

    aic_values = {}
    for name, copula in copulas.items():
        try:
            copula.fit(np.column_stack([u, v]))
            log_likelihood = calculate_log_likelihood_with_interaction(copula, u, v, x, y, tau1, tau2, gamma)
            num_params = copula.theta.size if isinstance(copula.theta, np.ndarray) else 1
            aic = calculate_aic(log_likelihood, num_params)

            print(f"{name} Copula Log-Likelihood (gamma={gamma}): {log_likelihood:.4f}")
            print(f"{name} Copula AIC (gamma={gamma}): {aic:.4f}")
            aic_values[name] = aic
        except Exception as e:
            print(f"Error fitting {name} Copula: {e}")
            aic_values[name] = float('inf')
    return aic_values

# Step 7: Plot Joint Distribution
def plot_joint_distribution(x, y, x_simulated, y_simulated, best_copula_name):
    sns.kdeplot(x=x, y=y, cmap='Blues', fill=True, alpha=0.7, label='Actual Data')
    sns.kdeplot(x=x_simulated, y=y_simulated, cmap='Reds', fill=True, alpha=0.7, label=f'{best_copula_name} Copula')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Step 8: Main Workflow
def main(file_path, gamma_values):
    cleaned_data = load_and_clean_data(file_path)
    x, y, tau1, tau2 = cleaned_data[:, 0], cleaned_data[:, 1], cleaned_data[:, 2], cleaned_data[:, 3]

    # Step 1: Fit Beta distributions
    alpha_x, beta_x = fit_beta_distribution(x)
    alpha_y, beta_y = fit_beta_distribution(y)

    # Step 2: Correlation analysis
    tau, p_value_tau = kendalltau(x, y)
    rho, p_value_rho = spearmanr(x, y)
    print(f"Kendall's tau: {tau:.4f}, p-value: {p_value_tau:.4e}")
    print(f"Spearman's rho: {rho:.4f}, p-value: {p_value_rho:.4e}")

    # Step 3: Transform data to marginal CDFs
    u = beta.cdf(x, alpha_x, beta_x)
    v = beta.cdf(y, alpha_y, beta_y)

    print(f"Fitted Beta Parameters for X: alpha = {alpha_x:.4f}, beta = {beta_x:.4f}")
    print(f"Fitted Beta Parameters for Y: alpha = {alpha_y:.4f}, beta = {beta_y:.4f}")

    # Step 4: Fit Copulas and calculate AIC for each gamma
    for gamma in gamma_values:
        print(f"\nEvaluating Copulas with gamma = {gamma}")
        aic_values = fit_copulas_and_calculate_aic(u, v, x, y, tau1.mean(), tau2.mean(), gamma)

        # Identify best Copula
        best_copula_name = min(aic_values, key=aic_values.get)
        print(f"Best Copula for gamma={gamma}: {best_copula_name} (AIC = {aic_values[best_copula_name]:.4f})")
        
        # Simulate data using the best Copula
        best_copula = {
            "Clayton": Clayton(),
            "Gumbel": Gumbel(),
            "Frank": Frank()
        }[best_copula_name]
        best_copula.fit(np.column_stack([u, v]))
        simulated_copula = best_copula.sample(len(x))

        # Transform simulated CDFs back to original scale
        x_simulated = beta.ppf(simulated_copula[:, 0], alpha_x, beta_x)
        y_simulated = beta.ppf(simulated_copula[:, 1], alpha_y, beta_y)

        # Apply interaction term
        x_simulated, y_simulated = apply_interaction_term(x_simulated, y_simulated, tau1.mean(), tau2.mean(), gamma)

        # Plot results for the best Copula
        plot_joint_distribution(x, y, x_simulated, y_simulated, best_copula_name)

# Execute Main Workflow
np.random.seed(81)
file_path = '/Users/lena/Desktop/lunwen/data/calculated_household_data_rural.xlsx'
gamma_values = [1, 5, 10, 20]
main(file_path, gamma_values)

"""
Total seed:81,94,111,120_Best Copula: Clayton (AIC = -0.4670)
Kendall's tau: 0.1481, p-value: 9.4705e-02
Spearman's rho: 0.2326, p-value: 7.3708e-02
Fitted Beta Parameters for X: alpha = 1.4657, beta = 11.4728
Fitted Beta Parameters for Y: alpha = 2.3667, beta = 4.9189

Evaluating Copulas with gamma = 1
Clayton Copula Log-Likelihood (gamma=1): 2.6272
Clayton Copula AIC (gamma=1): -3.2544
Gumbel Copula Log-Likelihood (gamma=1): -0.6550
Gumbel Copula AIC (gamma=1): 3.3100
Frank Copula Log-Likelihood (gamma=1): 1.0165
Frank Copula AIC (gamma=1): -0.0330
Best Copula for gamma=1: Clayton (AIC = -3.2544)

Evaluating Copulas with gamma = 5
Clayton Copula Log-Likelihood (gamma=5): 2.0078
Clayton Copula AIC (gamma=5): -2.0156
Gumbel Copula Log-Likelihood (gamma=5): -1.2745
Gumbel Copula AIC (gamma=5): 4.5489
Frank Copula Log-Likelihood (gamma=5): 0.3971
Frank Copula AIC (gamma=5): 1.2059
Best Copula for gamma=5: Clayton (AIC = -2.0156)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

Evaluating Copulas with gamma = 10
Clayton Copula Log-Likelihood (gamma=10): 1.2335
Clayton Copula AIC (gamma=10): -0.4670
Gumbel Copula Log-Likelihood (gamma=10): -2.0488
Gumbel Copula AIC (gamma=10): 6.0975
Frank Copula Log-Likelihood (gamma=10): -0.3772
Frank Copula AIC (gamma=10): 2.7545
Best Copula for gamma=10: Clayton (AIC = -0.4670)

Evaluating Copulas with gamma = 20
Clayton Copula Log-Likelihood (gamma=20): -0.3151
Clayton Copula AIC (gamma=20): 2.6303
Gumbel Copula Log-Likelihood (gamma=20): -3.5974
Gumbel Copula AIC (gamma=20): 9.1948
Frank Copula Log-Likelihood (gamma=20): -1.9258
Frank Copula AIC (gamma=20): 5.8517
Best Copula for gamma=20: Clayton (AIC = 2.6303)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

runfile('/Users/lena/Desktop/lunwen/python/code for submission/Final Copula Model.py', wdir='/Users/lena/Desktop/lunwen/python/code for submission')
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
Kendall's tau: -0.0870, p-value: 5.7223e-01
Spearman's rho: -0.1591, p-value: 4.5766e-01
Fitted Beta Parameters for X: alpha = 2.0997, beta = 25.3005
Fitted Beta Parameters for Y: alpha = 3.0427, beta = 5.0561

Evaluating Copulas with gamma = 1
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=1): 0.4634
Frank Copula AIC (gamma=1): 1.0733
Best Copula for gamma=1: Frank (AIC = 1.0733)

Evaluating Copulas with gamma = 5
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=5): 0.2727
Frank Copula AIC (gamma=5): 1.4545
Best Copula for gamma=5: Frank (AIC = 1.4545)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

Evaluating Copulas with gamma = 10
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=10): 0.0345
Frank Copula AIC (gamma=10): 1.9310
Best Copula for gamma=10: Frank (AIC = 1.9310)

Evaluating Copulas with gamma = 20
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=20): -0.4421
Frank Copula AIC (gamma=20): 2.8841
Best Copula for gamma=20: Frank (AIC = 2.8841)
"""
"""
Urban seed:71,95,102,104_Best Copula: Frank (AIC = 1.9310)
Kendall's tau: -0.0870, p-value: 5.7223e-01
Spearman's rho: -0.1591, p-value: 4.5766e-01
Fitted Beta Parameters for X: alpha = 2.0997, beta = 25.3005
Fitted Beta Parameters for Y: alpha = 3.0427, beta = 5.0561

Evaluating Copulas with gamma = 1
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=1): 0.4634
Frank Copula AIC (gamma=1): 1.0733
Best Copula for gamma=1: Frank (AIC = 1.0733)

Evaluating Copulas with gamma = 5
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=5): 0.2727
Frank Copula AIC (gamma=5): 1.4545
Best Copula for gamma=5: Frank (AIC = 1.4545)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

Evaluating Copulas with gamma = 10
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=10): 0.0345
Frank Copula AIC (gamma=10): 1.9310
Best Copula for gamma=10: Frank (AIC = 1.9310)

Evaluating Copulas with gamma = 20
Error fitting Clayton Copula: The computed theta value -0.16 is out of limits for the given CLAYTON copula.
Error fitting Gumbel Copula: The computed theta value 0.92 is out of limits for the given GUMBEL copula.
Frank Copula Log-Likelihood (gamma=20): -0.4421
Frank Copula AIC (gamma=20): 2.8841
Best Copula for gamma=20: Frank (AIC = 2.8841)
"""
"""
Rural seed:41,91,131,150_Best Copula: Frank (AIC = -12.2588)
Kendall's tau: 0.4095, p-value: 4.4110e-04
Spearman's rho: 0.6214, p-value: 5.2468e-05
Fitted Beta Parameters for X: alpha = 1.4548, beta = 9.1888
Fitted Beta Parameters for Y: alpha = 2.2278, beta = 5.4191

Evaluating Copulas with gamma = 1
Clayton Copula Log-Likelihood (gamma=1): 6.4918
Clayton Copula AIC (gamma=1): -10.9836
Gumbel Copula Log-Likelihood (gamma=1): 3.6606
Gumbel Copula AIC (gamma=1): -5.3212
Frank Copula Log-Likelihood (gamma=1): 7.6890
Frank Copula AIC (gamma=1): -13.3780
Best Copula for gamma=1: Frank (AIC = -13.3780)

Evaluating Copulas with gamma = 5
Clayton Copula Log-Likelihood (gamma=5): 6.2431
Clayton Copula AIC (gamma=5): -10.4862
Gumbel Copula Log-Likelihood (gamma=5): 3.4119
Gumbel Copula AIC (gamma=5): -4.8237
Frank Copula Log-Likelihood (gamma=5): 7.4403
Frank Copula AIC (gamma=5): -12.8806
Best Copula for gamma=5: Frank (AIC = -12.8806)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

Evaluating Copulas with gamma = 10
Clayton Copula Log-Likelihood (gamma=10): 5.9322
Clayton Copula AIC (gamma=10): -9.8644
Gumbel Copula Log-Likelihood (gamma=10): 3.1010
Gumbel Copula AIC (gamma=10): -4.2019
Frank Copula Log-Likelihood (gamma=10): 7.1294
Frank Copula AIC (gamma=10): -12.2588
Best Copula for gamma=10: Frank (AIC = -12.2588)
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

Evaluating Copulas with gamma = 20
Clayton Copula Log-Likelihood (gamma=20): 5.3104
Clayton Copula AIC (gamma=20): -8.6208
Gumbel Copula Log-Likelihood (gamma=20): 2.4792
Gumbel Copula AIC (gamma=20): -2.9583
No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
Frank Copula Log-Likelihood (gamma=20): 6.5076
Frank Copula AIC (gamma=20): -11.0152
Best Copula for gamma=20: Frank (AIC = -11.0152)
"""



