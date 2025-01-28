#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:17:09 2025

@author: lena
"""

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load data
file_path = '/Users/lena/Desktop/lunwen/data/data2.xlsx'
data_sheet1 = pd.read_excel(file_path, sheet_name='Malaria_Protection_Settlement')

# Extract the rows with non-empty Settlement columns
regions = data_sheet1.loc[data_sheet1['Settlement'].notnull(), 'Settlement'].reset_index(drop=True)

positive_households = data_sheet1.loc[data_sheet1['Malaria'] == 'Positive', 'Household'].reset_index(drop=True)
negative_households = data_sheet1.loc[data_sheet1['Malaria'] == 'Negative', 'Household'].reset_index(drop=True)

G1_protection = data_sheet1.loc[data_sheet1['Malaria'] == 'Positive', 'Protection'].reset_index(drop=True)
G2_protection = data_sheet1.loc[data_sheet1['Malaria'] == 'Negative', 'Protection'].reset_index(drop=True)

total_households = positive_households + negative_households

# Calculate tau_1, tau_2, x, y (relative to the total population N, same with our defination)
# x_prime, y_prime (relative to the population of each Group)
tau_1 = positive_households / total_households
tau_2 = negative_households / total_households
x_prime = G1_protection / positive_households
y_prime = G2_protection / negative_households
x = G1_protection / total_households
y = G2_protection / total_households

results = pd.DataFrame({
    'Region': regions,
    'Positive Households': positive_households,
    'Negative Households': negative_households,
    'Total Households': total_households,
    'G1 Protection': G1_protection,
    'G2 Protection': G2_protection,
    'Tau 1': tau_1,
    'Tau 2': tau_2,
    'x_prime': x_prime,
    'y_prime': y_prime,
    'x': x,
    'y': y
})

pd.set_option('display.max_columns', None)
print(results)
print(results.columns)

# Sva to Excel 
results.to_excel('/Users/lena/Desktop/lunwen/data/calculated_household_data.xlsx', index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions
plt.figure(figsize=(8, 6))
sns.histplot(tau_1, kde=True, color='orange', bins=20, label='Infection Rate (Tau_1 = N_1/N)')
plt.xlabel('Infection Rate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.savefig('Infection Rate Distribution.pdf', format='pdf', bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(x_prime, kde=True, color='blue', bins=20, label='Protection Rate (X_1/N_1 - Positive Group)')
plt.xlabel('Protection Rate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.savefig('Protection Rate Distribution of G1(with malaria).pdf', format='pdf', bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(y_prime, kde=True, color='green', bins=20, label='Protection Rate (X_2/N_2 - Negative Group)', alpha=0.5)
plt.xlabel('Protection Rate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.savefig('Protection Rate Distribution of G2(without malaria).pdf', format='pdf', bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(x_prime, kde=True, color='blue', bins=20, label='Protection Rate (X_1/N_1 - Positive Group)')
sns.histplot(y_prime, kde=True, color='green', bins=20, label='Protection Rate (X_2/N_2 - Negative Group)', alpha=0.5)
plt.xlabel('Protection Rate')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.savefig('Protection Rate Distribution.pdf', format='pdf', bbox_inches="tight")
plt.show()

# A trial for scatter plot (rural)
file_path = '/Users/lena/Desktop/lunwen/data/calculated_household_data_rural.xlsx'
results_2=pd.read_excel(file_path)

x = results_2['x']  # Protection Rate (x)
tau_1 = results_2['Tau 1']  # Infection Rate (tau1)
regions = results_2['Region']

plt.figure(figsize=(10, 6))
plt.scatter(x, tau_1, color='blue', alpha=0.7)

for i, region in enumerate(regions):
    plt.text(x[i], tau_1[i], region, fontsize=8, alpha=0.7)

plt.title('Scatter Plot: Protection Rate (x) vs Infection Rate (tau1)')
plt.xlabel('Protection Rate (x - Positive)')
plt.ylabel('Infection Rate (tau1)')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

