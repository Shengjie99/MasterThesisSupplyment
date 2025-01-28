#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:20:23 2025

@author: lena
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# Part1
# 1. Data loading and cleaning
def load_and_clean_data(file_path):
    """
    Load Excel file and perform data cleaning:
    - Remove invalid data for x <= 0, y <= 0, tau1 <= x, tau2 <= y.
    - Extract the Positive Households column as the I value.
    """
    data = pd.read_excel(file_path)
    # Extract x, y, Tau 1, Tau 2 and Positive Households (I)
    data_points = data[['x', 'y', 'Tau 1', 'Tau 2', 'Positive Households']].dropna().values
    cleaned_data = data_points[
        (data_points[:, 0] > 0) & (data_points[:, 1] > 0) &
        (data_points[:, 2] > data_points[:, 0]) & (data_points[:, 3] > data_points[:, 1])
    ]
    return cleaned_data[:, :4], cleaned_data[:, 4]

# 2. Calculate the normalization constant C with I correction
def compute_C_stable(tau1, tau2, gamma, Z1_plus, Z1_minus, Z2_plus, Z2_minus, I, Z1_plus_prime):
    def integrand(x, y):
        # Adjust Z1_plus by adding I * Z1_plus_prime
        Z1_plus_effective = Z1_plus + I * Z1_plus_prime
        return (x**(Z1_plus_effective - 1)) * ((tau1 - x)**(Z1_minus - 1)) * \
               (y**(Z2_plus - 1)) * ((tau2 - y)**(Z2_minus - 1)) * \
               np.exp(-gamma / 2 * (tau2 * x - tau1 * y)**2)

    integral, _ = dblquad(
        lambda y, x: integrand(x, y),
        1e-5, tau1 - 1e-5,
        lambda x: 1e-5, lambda x: tau2 - 1e-5
    )
    return 1 / integral  

# 3. Modified log-likelihood function (with I correction)
def log_likelihood_with_incidence(params, data, incidence_cases):
    """
    Modified log-likelihood function with Z1_plus replaced by Z1_plus + I * Z1_plus_prime
    :param params: model parameters
    :param data: cleaned data
    :param incidence_cases: I values for each data set (known)
    """
    gamma, Z1_plus, Z1_plus_prime, Z1_minus, Z2_plus, Z2_minus = params
    log_lik = 0
    for i, (x, y, tau1, tau2) in enumerate(data):
        I = incidence_cases[i]
        Z1_plus_effective = Z1_plus + I * Z1_plus_prime  # Dynamic calculations for Z1_plus with I
        C = compute_C_stable(tau1, tau2, gamma, Z1_plus, Z1_minus, Z2_plus, Z2_minus, I, Z1_plus_prime)
        prob_density = C * (x**(Z1_plus_effective - 1)) * ((tau1 - x)**(Z1_minus - 1)) * \
                       (y**(Z2_plus - 1)) * ((tau2 - y)**(Z2_minus - 1)) * \
                       np.exp(-gamma / 2 * (tau2 * x - tau1 * y)**2)
        log_lik += np.log(prob_density + 1e-10)  # Avoid log(0)
    return -log_lik

# 4. AIC Cal
def calculate_aic(neg_log_likelihood, num_params):
    """
    Calculate the AIC value
    :param neg_log_likelihood: negative log likelihood value
    :param num_params: number of model parameters
    :return: AIC value
    """
    return 2 * num_params + 2 * neg_log_likelihood

def compute_kl_divergence(p, q):
    """
    Compute Kullback-Leibler (KL) divergence between two probability distributions p and q.
    Both p and q must be normalized and have the same length.
    """
    p = np.array(p) / np.sum(p)  # Ensure p is a probability distribution
    q = np.array(q) / np.sum(q)  # Ensure q is a probability distribution
    return entropy(p, q)  # KL divergence using scipy

def compute_js_divergence(p, q):
    """
    Compute Jensen-Shannon (JS) divergence between two probability distributions p and q.
    Both p and q must be normalized and have the same length.
    """
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    m = 0.5 * (p + q)  # Average distribution
    return 0.5 * (entropy(p, m) + entropy(q, m))

# 5. Main Function
def main(file_path):
    cleaned_data, incidence_cases = load_and_clean_data(file_path)
    print(f"Number of Cleaned Data: {len(cleaned_data)}")
    
    # Step 2: Define initial parameters and boundaries
    initial_params_with_incidence = [1.0, 2.0, 1.0, 2.0, 2.0, 2.0]  # gamma, Z1_plus, Z1_plus_prime, Z1_minus, Z2_plus, Z2_minus
    bounds_with_incidence = [(1e-3, 300), (0.001, 10), (0, 5), (0.001, 10), (0.001, 10), (0.001, 10)]
    
    # Step 3: Execution optimization
    result_with_incidence = minimize(
        log_likelihood_with_incidence,
        initial_params_with_incidence,
        args=(cleaned_data, incidence_cases),
        method='trust-constr',
        bounds=bounds_with_incidence
    )
    
    optimal_params = result_with_incidence.x
    gamma, Z1_plus, Z1_plus_prime, Z1_minus, Z2_plus, Z2_minus = optimal_params
    print("Optimization results (with I correction):")
    print(f"gamma: {gamma:.4f}, Z1_plus: {Z1_plus:.4f}, Z1_plus_prime: {Z1_plus_prime:.4f}, "
          f"Z1_minus: {Z1_minus:.4f}, Z2_plus: {Z2_plus:.4f}, Z2_minus: {Z2_minus:.4f}")
    
    # Step 5: AIC
    neg_log_likelihood = log_likelihood_with_incidence(optimal_params, cleaned_data, incidence_cases)
    aic = calculate_aic(neg_log_likelihood, num_params=6)
    print(f"AIC of Model: {aic:.4f}")
    
    # Compute KL and JS divergence
    actual_probabilities = []
    fitted_probabilities = []
    for i, (x, y, tau1, tau2) in enumerate(cleaned_data):
        I = incidence_cases[i]
        Z1_plus_effective = Z1_plus + I * Z1_plus_prime  # Dynamic calculations for Z1_plus with I
        C = compute_C_stable(tau1, tau2, gamma, Z1_plus, Z1_minus, Z2_plus, Z2_minus, I, Z1_plus_prime)
        fitted_prob = C * (x**(Z1_plus_effective - 1)) * ((tau1 - x)**(Z1_minus - 1)) * \
                      (y**(Z2_plus - 1)) * ((tau2 - y)**(Z2_minus - 1)) * \
                      np.exp(-gamma / 2 * (tau2 * x - tau1 * y)**2)
        fitted_probabilities.append(fitted_prob)
        actual_probabilities.append(1)  # Assuming each data point contributes equally

    kl_div = compute_kl_divergence(actual_probabilities, fitted_probabilities)
    js_div = compute_js_divergence(actual_probabilities, fitted_probabilities)
    print(f"KL Divergence: {kl_div:.4f}, JS Divergence: {js_div:.4f}")
    
    return cleaned_data, optimal_params, kl_div, js_div

# Execute the main function
file_path = '/Users/lena/Desktop/lunwen/data/calculated_household_data.xlsx'  # Replace with your data path
cleaned_data, optimal_params, kl_div, js_div = main(file_path)


# Plotting the variation of the normalization constant C with gamma
test_gammas = [0.1, 1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


Cs = [compute_C_stable(tau1=2, tau2=3, gamma=gamma, Z1_plus=2, Z1_minus=2, Z2_plus=2, Z2_minus=2, 
                        I=1, Z1_plus_prime=0.5) for gamma in test_gammas]  # 假设 I=1 和 Z1_plus_prime=0.5，修改为实际值

plt.plot(test_gammas, Cs, marker='o')
plt.xlabel('gamma')
plt.ylabel('C (Normalization Constant)')
plt.grid()
plt.savefig('C_vs_gamma.pdf', format='pdf', bbox_inches="tight")
plt.show()

# Plotting the log-likelihood as a function of gamma
gammas = np.linspace(0.1, 1000, 50)

# incidence_cases
log_likelihoods = [
    log_likelihood_with_incidence([gamma, 2, 0.5, 2, 2, 2], cleaned_data, np.ones(cleaned_data.shape[0])) 
    for gamma in gammas
]

plt.plot(gammas, log_likelihoods)
plt.xlabel('gamma')
plt.title('Log Likelihood vs Gamma')
plt.grid()
plt.savefig('Log_Likelihood_vs_Gamma.pdf', format='pdf', bbox_inches="tight")
plt.show()

# Calculate the range of tau2*x - tau1*y
tau2_x_tau1_y = cleaned_data[:, 3] * cleaned_data[:, 0] - cleaned_data[:, 2] * cleaned_data[:, 1]
print(f"Min of tau2*x - tau1*y: {np.min(tau2_x_tau1_y)}")
print(f"Max of tau2*x - tau1*y: {np.max(tau2_x_tau1_y)}")
print(f"Mean of tau2*x - tau1*y: {np.mean(tau2_x_tau1_y)}")

sns.histplot(tau2_x_tau1_y, kde=True, bins=30)
plt.xlabel('tau2 * x - tau1 * y')
plt.ylabel('Frequency')
plt.savefig('Distribution_of_tau2_x_tau1_y.pdf', format='pdf', bbox_inches="tight")
plt.show()

# Plot the range of (tau2*x - tau1*y)^2
tau2_x_tau1_y_squared = (cleaned_data[:, 3] * cleaned_data[:, 0] - cleaned_data[:, 2] * cleaned_data[:, 1])**2
sns.histplot(tau2_x_tau1_y_squared, kde=True, bins=30)
plt.xlabel('(tau2*x - tau1*y)^2')
plt.ylabel('Frequency')
plt.savefig('Distribution_of_tau2_x_tau1_y_squared.pdf', format='pdf', bbox_inches="tight")
plt.show()

### Plotting the kernel density of the fitted u(x, y) distribution
x_vals = np.linspace(0.01, 0.99, 100)
y_vals = np.linspace(0.01, 0.99, 100)
X, Y = np.meshgrid(x_vals, y_vals)

gamma, Z1_plus, Z1_plus_prime, Z1_minus, Z2_plus, Z2_minus = optimal_params
tau1 = cleaned_data[:, 2].mean()
tau2 = cleaned_data[:, 3].mean()

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = (
            (X[i, j]**(Z1_plus - 1)) * ((tau1 - X[i, j])**(Z1_minus - 1)) *
            (Y[i, j]**(Z2_plus - 1)) * ((tau2 - Y[i, j])**(Z2_minus - 1)) *
            np.exp(-gamma / 2 * (tau2 * X[i, j] - tau1 * Y[i, j])**2)
        )

# Kernel density map of actual data
x = cleaned_data[:, 0]
y = cleaned_data[:, 1]
plt.figure(figsize=(10, 8))
sns.kdeplot(x=x, y=y, cmap='Blues', fill=True, alpha=0.7, label='Actual Data')

# Kernel density mapping using the generated Z data
sns.kdeplot(x=X.ravel(), y=Y.ravel(), weights=Z.ravel(), cmap='Reds', fill=True, alpha=0.7, label='Fitted u(x, y)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Actual Data vs u(x, y).pdf', format='pdf', bbox_inches="tight")
plt.show()

### Plotting fitted u(x, y) contours and histograms of actual data
plt.figure(figsize=(10, 8))
sns.histplot(x=x, y=y, bins=30, pmax=0.8, cmap='Blues', cbar=True, label='Actual Data')

contour = plt.contour(X, Y, Z, levels=10, cmap='Reds', alpha=0.7)
plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

x_margin = 0.5  # 5% margin to the right

# Adjust x and y axis limits with a little padding
plt.xlim(min(x), max(x) * (1 + x_margin))  # Slight extension on the right

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Actual Data (Bar)_u(x, y) (Fitted Contour).pdf', format='pdf', bbox_inches="tight")
plt.show()


"""
total data
gamma: 10.0000, Z1_plus: 1.4758, Z1_plus_prime: 0.0172, Z1_minus: 2.6077, Z2_plus: 2.2278, Z2_minus: 2.8183
AIC of Model: -256.7323
KL Divergence: 0.2758, JS Divergence: 0.0547


gamma: 50.0000, Z1_plus: 1.4161, Z1_plus_prime: 0.0172, Z1_minus: 2.5094, Z2_plus: 2.1540, Z2_minus: 2.7167
AIC of Model: -262.5192
KL Divergence: 0.2474, JS Divergence: 0.0496


gamma: 100.0000, Z1_plus: 1.3551, Z1_plus_prime: 0.0171, Z1_minus: 2.4089, Z2_plus: 2.0800, Z2_minus: 2.6140
AIC of Model: -268.9639
KL Divergence: 0.2195, JS Divergence: 0.0444

gamma: 200.0000, Z1_plus: 1.2664, Z1_plus_prime: 0.0169, Z1_minus: 2.2639, Z2_plus: 1.9769, Z2_minus: 2.4681
AIC of Model: -279.6839
KL Divergence: 0.1815, JS Divergence: 0.0372

gamma: 300.0000, Z1_plus: 1.2064, Z1_plus_prime: 0.0168, Z1_minus: 2.1677, Z2_plus: 1.9128, Z2_minus: 2.3731
AIC of Model: -288.1894
KL Divergence: 0.1584, JS Divergence: 0.0328
"""


"""
urban data

gamma: 10.0000, Z1_plus: 3.2451, Z1_plus_prime: 0.0000, Z1_minus: 4.5231, Z2_plus: 3.6286, Z2_minus: 4.3325
AIC of Model: -124.8049

gamma: 50.0000, Z1_plus: 3.1834, Z1_plus_prime: 0.0000, Z1_minus: 4.4462, Z2_plus: 3.5698, Z2_minus: 4.2428
AIC of Model: -125.6255

gamma: 100.0000, Z1_plus: 3.1140, Z1_plus_prime: 0.0000, Z1_minus: 4.3606, Z2_plus: 3.5046, Z2_minus: 4.1419
AIC of Model: -126.5721

gamma: 200.0000, Z1_plus: 2.9960, Z1_plus_prime: 0.0000, Z1_minus: 4.2179, Z2_plus: 3.3960, Z2_minus: 3.9704
AIC of Model: -128.2379

gamma: 300.0000, Z1_plus: 2.8991, Z1_plus_prime: 0.0000, Z1_minus: 4.1042, Z2_plus: 3.3098, Z2_minus: 3.8296
AIC of Model: -129.6522

gamma: 400.0000, Z1_plus: 2.8177, Z1_plus_prime: 0.0000, Z1_minus: 4.0120, Z2_plus: 3.2399, Z2_minus: 3.7113
AIC of Model: -130.8620

gamma: 500.0000, Z1_plus: 2.7482, Z1_plus_prime: 0.0000, Z1_minus: 3.9361, Z2_plus: 3.1823, Z2_minus: 3.6101
AIC of Model: -131.9026

"""


"""
rural data
gamma: 10.0000, Z1_plus: 0.7794, Z1_plus_prime: 0.0261, Z1_minus: 2.2460, Z2_plus: 1.7790, Z2_minus: 2.3297
AIC of Model: -132.0788

gamma: 50.0000, Z1_plus: 0.7232, Z1_plus_prime: 0.0260, Z1_minus: 2.1366, Z2_plus: 1.6994, Z2_minus: 2.2245
AIC of Model: -137.3639

gamma: 200.0000, Z1_plus: 0.6002, Z1_plus_prime: 0.0258, Z1_minus: 1.8841, Z2_plus: 1.5309, Z2_minus: 2.0113
AIC of Model: -152.3466

gamma: 300.0000, Z1_plus: 0.5595, Z1_plus_prime: 0.0258, Z1_minus: 1.7947, Z2_plus: 1.4827, Z2_minus: 1.9554
AIC of Model: -159.3445

gamma: 400.0000, Z1_plus: 0.5330, Z1_plus_prime: 0.0258, Z1_minus: 1.7346, Z2_plus: 1.4574, Z2_minus: 1.9290
AIC of Model: -164.8260

gamma: 500.0000, Z1_plus: 0.5141, Z1_plus_prime: 0.0258, Z1_minus: 1.6905, Z2_plus: 1.4443, Z2_minus: 1.9179
AIC of Model: -169.2404
"""














