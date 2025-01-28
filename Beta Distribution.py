#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:16:53 2025

@author: lena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize
import datetime

# Load and clean data
def load_and_clean_data(file_path):
    """
    Load the data and clean it for Beta fitting.
    - Remove zero values to ensure Beta distribution validity.
    """
    data = pd.read_excel(file_path)
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("The input file must contain 'x' and 'y' columns.")
    
    # Remove NaN and zero values, and ensure values are strictly in (0, 1)
    data_points = data[['x', 'y']].dropna()
    data_points = data_points[(data_points['x'] > 0) & (data_points['y'] > 0)]
    data_points['x'] = np.clip(data_points['x'], 1e-6, 1 - 1e-6)
    data_points['y'] = np.clip(data_points['y'], 1e-6, 1 - 1e-6)

    print("First few rows of the cleaned data (after removing zeros):")
    print(data_points.head())
    return data_points

# Check for overdispersion
def check_overdispersion(data):
    """
    Check for overdispersion by comparing variance and mean for x and y.
    """
    mean_x, var_x = np.mean(data['x']), np.var(data['x'], ddof=1)
    mean_y, var_y = np.mean(data['y']), np.var(data['y'], ddof=1)

    ratio_x = var_x / mean_x if mean_x > 0 else np.inf
    ratio_y = var_y / mean_y if mean_y > 0 else np.inf

    print("\nOverdispersion Check:")
    print(f"x - Mean: {mean_x:.4f}, Variance: {var_x:.4f}, Variance-to-Mean Ratio: {ratio_x:.4f}")
    print(f"y - Mean: {mean_y:.4f}, Variance: {var_y:.4f}, Variance-to-Mean Ratio: {ratio_y:.4f}")

    # Determine if overdispersion is present
    if ratio_x > 1:
        print("x shows overdispersion.")
    else:
        print("x does not show overdispersion.")
    
    if ratio_y > 1:
        print("y shows overdispersion.")
    else:
        print("y does not show overdispersion.")

# Negative log-likelihood for Beta distribution
def neg_log_likelihood(params, data):
    """
    Negative log-likelihood function for Beta distribution.
    :param params: [alpha, beta]
    :param data: observed data
    """
    alpha, beta_param = params
    if alpha <= 0 or beta_param <= 0:
        return np.inf  # Penalize invalid parameters
    log_likelihood = np.sum(beta.logpdf(data, alpha, beta_param))
    return -log_likelihood

# Fit Beta distribution
def fit_beta_distribution(data, column_name):
    """
    Fit Beta distribution to the specified column.
    """
    column_data = data[column_name]
    initial_params = [1.0, 1.0]  # Initial guesses for alpha and beta
    bounds = [(1e-3, None), (1e-3, None)]  # Ensure parameters are positive

    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(column_data,),
        method='L-BFGS-B',
        bounds=bounds
    )

    if result.success:
        alpha, beta_param = result.x
        # Calculate AIC and BIC
        n = len(column_data)
        log_likelihood = -neg_log_likelihood(result.x, column_data)
        aic = 2 * len(result.x) - 2 * log_likelihood
        bic = len(result.x) * np.log(n) - 2 * log_likelihood

        print(f"\nFitted Beta parameters for {column_name}: alpha = {alpha:.4f}, beta = {beta_param:.4f}")
        print(f"Log-Likelihood: {log_likelihood:.4f}")
        print(f"AIC: {aic:.4f}, BIC: {bic:.4f}")
        return alpha, beta_param, aic, bic
    else:
        raise ValueError(f"Beta distribution fitting failed for {column_name}: {result.message}")

# Plot actual vs fitted Beta distribution
def plot_actual_vs_fitted(data, column_name, alpha, beta_param):
    """
    Plot histograms of actual data and fitted Beta distribution.
    """
    column_data = data[column_name]
    x_vals = np.linspace(0, 1, 100)  # Generate x-axis values for Beta distribution
    fitted_pdf = beta.pdf(x_vals, alpha, beta_param)

    # Plot histogram of actual data and fitted distribution
    plt.figure(figsize=(10, 6))
    plt.hist(column_data, bins=30, density=True, alpha=0.6, label="Actual Data", color='blue')
    plt.plot(x_vals, fitted_pdf, 'r-', label="Fitted Beta Distribution", linewidth=2)
    plt.xlabel(column_name)
    plt.ylabel("Density")
    plt.title(f"Actual vs Fitted Beta Distribution ({column_name})")
    plt.legend()
    plt.grid()
    plt.show()


# Plot actual vs fitted Beta distribution
def plot_actual_vs_fitted(data, column_name, alpha, beta_param, file_path=None):
    """
    Plot histograms of actual data and fitted Beta distribution, and save the plot as a PDF.
    
    Parameters:
    - data: DataFrame containing the data.
    - column_name: str, name of the column to plot.
    - alpha: float, alpha parameter of the Beta distribution.
    - beta_param: float, beta parameter of the Beta distribution.
    - file_path: str, optional, base file path to save the PDF file. A timestamp will be added to ensure uniqueness.
    """
    column_data = data[column_name]
    x_vals = np.linspace(0, 1, 100)  # Generate x-axis values for Beta distribution
    fitted_pdf = beta.pdf(x_vals, alpha, beta_param)

    # Plot histogram of actual data and fitted distribution
    plt.figure(figsize=(10, 6))
    plt.hist(column_data, bins=30, density=True, alpha=0.6, label="Actual Data", color='blue')
    plt.plot(x_vals, fitted_pdf, 'r-', label="Fitted Beta Distribution", linewidth=2)
    plt.xlabel(column_name)
    plt.ylabel("Density")
    plt.legend()
    plt.grid()

    # Generate a unique file name with timestamp
    if file_path:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{file_path}_{timestamp}.pdf"
    else:
        file_name = f"fitted_beta_distribution_{column_name}.pdf"

    # Save the plot as a PDF file
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {file_name}")
    plt.show()

# Main function
def main(file_path):
    # Load and clean data
    data = load_and_clean_data(file_path)

    # Check for overdispersion
    check_overdispersion(data)

    # Fit Beta distribution for X
    alpha_x, beta_x, aic_x, bic_x = fit_beta_distribution(data, 'x')
    plot_actual_vs_fitted(data, 'x', alpha_x, beta_x)

    # Fit Beta distribution for Y
    alpha_y, beta_y, aic_y, bic_y = fit_beta_distribution(data, 'y')
    plot_actual_vs_fitted(data, 'y', alpha_y, beta_y)

    return (alpha_x, beta_x, aic_x, bic_x), (alpha_y, beta_y, aic_y, bic_y)

# Execute main function
file_path = '/Users/lena/Desktop/lunwen/data/calculated_household_data_rural.xlsx'  # Replace with your data path
try:
    (x_params, y_params) = main(file_path)
except Exception as e:
    print(f"An error occurred: {e}")

"""
total
x - Mean: 0.1104, Variance: 0.0073, Variance-to-Mean Ratio: 0.0662
y - Mean: 0.3434, Variance: 0.0339, Variance-to-Mean Ratio: 0.0988
x does not show overdispersion.
y does not show overdispersion.

Fitted Beta parameters for x: alpha = 1.5103, beta = 12.1395
Log-Likelihood: 81.4088
AIC: -158.8177, BIC: -154.4999

Fitted Beta parameters for y: alpha = 2.1653, beta = 4.0910
Log-Likelihood: 23.1840
AIC: -42.3680, BIC: -38.0502
"""
"""
urban
x - Mean: 0.0750, Variance: 0.0030, Variance-to-Mean Ratio: 0.0401
y - Mean: 0.4147, Variance: 0.0353, Variance-to-Mean Ratio: 0.0852
x does not show overdispersion.
y does not show overdispersion.

Fitted Beta parameters for x: alpha = 2.3342, beta = 28.5737
Log-Likelihood: 49.5820
AIC: -95.1639, BIC: -92.4995

Fitted Beta parameters for y: alpha = 2.7080, beta = 3.8117
Log-Likelihood: 9.0320
AIC: -14.0640, BIC: -11.3995

"""
"""
rural
x - Mean: 0.1380, Variance: 0.0091, Variance-to-Mean Ratio: 0.0657
y - Mean: 0.2879, Variance: 0.0266, Variance-to-Mean Ratio: 0.0923
x does not show overdispersion.
y does not show overdispersion.

Fitted Beta parameters for x: alpha = 1.4548, beta = 9.1888
Log-Likelihood: 38.1463
AIC: -72.2925, BIC: -69.1255

Fitted Beta parameters for y: alpha = 2.2278, beta = 5.4191
Log-Likelihood: 18.2388
AIC: -32.4775, BIC: -29.3105

"""