# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nbinom
from statsmodels.tools import add_constant
from statsmodels.discrete.count_model import NegativeBinomialP

# Part 1: Load and clean data
def load_and_clean_data(file_path):
    """
    Load and clean data for overdispersion check and NB fitting.
    """
    data = pd.read_excel(file_path)
    data_points = data[['x', 'y']].dropna()
    return data_points

# Part 2: Check for overdispersion
def check_overdispersion(data):
    """
    Check if data exhibits overdispersion by comparing variance and mean.
    """
    mean_x = np.mean(data['x'])
    var_x = np.var(data['x'], ddof=1)  # Sample variance
    
    mean_y = np.mean(data['y'])
    var_y = np.var(data['y'], ddof=1)

    print("Overdispersion Check:")
    print(f"x - Mean: {mean_x:.4f}, Variance: {var_x:.4f}, Variance-to-Mean Ratio: {var_x / mean_x:.4f}")
    print(f"y - Mean: {mean_y:.4f}, Variance: {var_y:.4f}, Variance-to-Mean Ratio: {var_y / mean_y:.4f}")

    return (var_x / mean_x, var_y / mean_y)

# Part 3: Fit Negative Binomial Model
def fit_negative_binomial(data):
    X = add_constant(data['x'])
    y = data['y']  

    # NegativeBinomialP
    model = NegativeBinomialP(y, X)
    nb_model = model.fit()

    print("Negative Binomial P Model Summary:")
    print(nb_model.summary())
    return nb_model

# Part 4: Plot actual vs fitted data
def plot_actual_vs_fitted(data, nb_model):
    """
    Plot histograms of actual and fitted distributions.
    """
    actual_counts = data['y'].value_counts(normalize=True).sort_index()
    x_vals = actual_counts.index  # Use the actual counts range

    # Predicted mean and parameters
    predicted_means = nb_model.predict(add_constant(data['x']))
    alpha = nb_model.params.get('alpha', 1.0)
    p = alpha / (alpha + predicted_means)

    # Generate fitted probabilities
    fitted_probs = nbinom.pmf(x_vals, alpha, p)

    plt.figure(figsize=(10, 6))

    sns.histplot(data['y'], bins=30, kde=False, color='blue', label='Actual Data', stat='density', alpha=0.6)
    plt.plot(x_vals, fitted_probs, color='red', label='Fitted Negative Binomial', linewidth=2)

    plt.xlabel('Counts')
    plt.ylabel('Density')
    plt.title('Actual vs Fitted Distributions')
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main(file_path):
    # Step 1: Load and clean data
    data = load_and_clean_data(file_path)
    
    # Step 2: Check for overdispersion
    overdispersion_ratios = check_overdispersion(data)

    # Step 3: Fit Negative Binomial Model
    nb_model = fit_negative_binomial(data)

    # Step 4: Plot actual vs fitted data
    plot_actual_vs_fitted(data, nb_model)

    print(f"Model AIC: {nb_model.aic:.4f}")

    return nb_model, overdispersion_ratios

# Execute the main function
file_path = '/Users/lena/Desktop/lunwen/data/calculated_household_data.xlsx'  # Replace with your data path
nb_model, overdispersion_ratios = main(file_path)


