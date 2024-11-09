# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:49:47 2024

@author: Hp
"""


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots and global figure size
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the data
data_path = r'F:/USD/Projects/Python_Diwali_Sales_Analysis-main/Diwali Sales Data.csv'
df = pd.read_csv(data_path, encoding='unicode_escape')

# Basic inspection of the dataset
print("Dataset shape:", df.shape)
print("First 10 rows:\n", df.head(10))
df.info()  # Shows column names, data types, and missing values

# Drop unnecessary columns safely (only if they exist in the data)
columns_to_drop = ['Status', 'unnamed1']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
print("Remaining columns:", df.columns)

# Drop rows with missing values and check the new shape of the dataset
print("Null values per column before dropping:\n", df.isnull().sum())
df.dropna(inplace=True)
print("Dataset shape after dropping null values:", df.shape)

# Convert 'Amount' column to integer if it exists
if 'Amount' in df.columns:
    df['Amount'] = df['Amount'].astype('int')

# Rename 'Marital_Status' to 'Shaadi' for easier understanding
df.rename(columns={'Marital_Status': 'Shaadi'}, inplace=True)

# Display descriptive statistics
print("\nDataset description:\n", df.describe())

# Define a reusable function to create a bar plot with data labels
def bar_plot_with_labels(data, x, y, title, xlabel, ylabel, rotation=0):
    """
    Creates a bar plot with data labels.

    Parameters:
    - data (DataFrame): Data for the plot.
    - x (str): Column name for x-axis.
    - y (str): Column name for y-axis.
    - title (str): Title of the plot.
    - xlabel (str): Label for x-axis.
    - ylabel (str): Label for y-axis.
    - rotation (int): Rotation angle for x-axis labels.
    """
    ax = sns.barplot(data=data, x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation)
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.show()

# Plotting the distribution of Gender
def plot_gender_distribution():
    """
    Plots the distribution of 'Gender' in the dataset.
    """
    ax = sns.countplot(x='Gender', data=df)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0])
    plt.show()

plot_gender_distribution()

# Plotting total Amount by Gender
if 'Gender' in df.columns and 'Amount' in df.columns:
    sales_gen = df.groupby('Gender', as_index=False)['Amount'].sum()
    bar_plot_with_labels(sales_gen, 'Gender', 'Amount', "Total Amount by Gender", "Gender", "Total Amount")

# Plotting Age Group distribution by Gender
def plot_age_group_distribution():
    """
    Plots the distribution of 'Age Group' by 'Gender'.
    """
    if 'Age Group' in df.columns and 'Gender' in df.columns:
        ax = sns.countplot(data=df, x='Age Group', hue='Gender')
        ax.set_title("Age Group Distribution by Gender")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Count")
        ax.bar_label(ax.containers[0])
        plt.show()

plot_age_group_distribution()

# Plotting total Amount by Age Group
if 'Age Group' in df.columns and 'Amount' in df.columns:
    sales_age = df.groupby('Age Group', as_index=False)['Amount'].sum()
    bar_plot_with_labels(sales_age, 'Age Group', 'Amount', "Total Amount by Age Group", "Age Group", "Total Amount")

# Top 15 states by number of orders
if 'State' in df.columns and 'Orders' in df.columns:
    top_states_orders = df.groupby('State', as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(15)
    bar_plot_with_labels(top_states_orders, 'State', 'Orders', "Top 15 States by Number of Orders", "State", "Number of Orders", rotation=45)

# Top 15 states by total Amount
if 'State' in df.columns and 'Amount' in df.columns:
    top_states_amount = df.groupby('State', as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(15)
    bar_plot_with_labels(top_states_amount, 'State', 'Amount', "Top 15 States by Total Amount", "State", "Total Amount", rotation=45)

# Number of orders by occupation
def plot_orders_by_occupation():
    """
    Plots the number of orders by 'Occupation'.
    """
    if 'Occupation' in df.columns:
        ax = sns.countplot(data=df, x='Occupation')
        ax.set_title("Number of Orders by Occupation")
        ax.set_xlabel("Occupation")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        ax.bar_label(ax.containers[0])
        plt.show()

plot_orders_by_occupation()

# Total amount by Occupation
if 'Occupation' in df.columns and 'Amount' in df.columns:
    sales_occ = df.groupby('Occupation', as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)
    bar_plot_with_labels(sales_occ, 'Occupation', 'Amount', "Total Amount by Occupation", "Occupation", "Total Amount", rotation=45)

# Top 15 most sold products by Product_ID
if 'Product_ID' in df.columns and 'Orders' in df.columns:
    top_products = df.groupby('Product_ID', as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(15)
    bar_plot_with_labels(top_products, 'Product_ID', 'Orders', "Top 15 Most Sold Products by Product_ID", "Product ID", "Number of Orders", rotation=45)
