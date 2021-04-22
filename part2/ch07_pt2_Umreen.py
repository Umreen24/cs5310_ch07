#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:38:12 2021

@author: umreenimam
"""

"""""""""""""""
IMPORTING PACKAGES
"""""""""""""""
import os
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, minmax_scale

"""""""""""""""
FUNCTIONS
"""""""""""""""
# Function to read and load data
def load_read_data(data):
    loaded_data = pd.read_excel(data, engine = 'openpyxl')
    
    return loaded_data

# Function to remove columns
def remove_cols(data):
    proteins = data.iloc[:, 2]
    data = data.iloc[:, 10:32]
    data_t = data.transpose()
    renamed_df = data_t.rename(columns = proteins)
    
    return renamed_df

# Zero fill-in rows function 
def zero_fill(data):
    filled_in_data = data.replace('Filtered', 0.0)
    
    return filled_in_data

# Remove co-linearity
def remove_colinearity(data, neg_threshold):
    corr_mat = data.corr()
    row = corr_mat.shape[0]
    column = corr_mat.shape[1]
    
    correlated_features = []
    
    for x in range(row): 
        for y in range(column):
            if x == y:
                break
            if corr_mat.iloc[x, y] > abs(neg_threshold) or corr_mat.iloc[x, y] < neg_threshold:
                correlated_features.append(corr_mat.columns[x])
                break

    return corr_mat, correlated_features

# Normalize data using min max scaler
def min_max(dataframe, array): 
    scaler = MinMaxScaler()
    normalized_corr = scaler.fit_transform(dataframe)
    normalied_corr_df = pd.DataFrame(normalized_corr)
    
    array_scaled = minmax_scale(array)
    
    return normalied_corr_df, array_scaled

def data_precision_recall(conf_mat):
    true_pos = np.diag(conf_mat)
    precision = np.mean(true_pos / np.sum(conf_mat, axis = 0))
    recall = np.mean(true_pos / np.sum(conf_mat, axis = 1))
    
    return precision, recall

"""""""""""""""
Q1-Q2: Load data into dataframe and extract "TBUT (sec) 0=1-5, 1=5-10, 2=>10"
"""""""""""""""
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/chapter07/lab/part2')
filename = 'PatientInfo.xlsx'
data_df = load_read_data(filename)

# Extract "TBUT (sec) 0=1-5, 1=5-10, 2=>10" column
# Use .values property to extract column into array
tbut_values = data_df.iloc[:, 24].values

"""""""""""""""
Q3: Load protein data into dataframe
"""""""""""""""
protein_filename = 'Data_SubjectStripTearSamples-Dec2020_Report.xlsx'
protein_data_df = load_read_data(protein_filename)

"""""""""""""""
Q4 & Q6: Remove unecessary columns and transpose dataframe
"""""""""""""""
protein_data = remove_cols(protein_data_df)

"""""""""""""""
Q5: Zero fill-in rows with 'filtered' in the cells
"""""""""""""""
zero_filled_df = zero_fill(protein_data)
zero_filled_df.astype(float)

"""""""""""""""
Q7-Q8: Remove co-linearity & normalize remaining column
"""""""""""""""
correlation_matrix, cols_to_remove = remove_colinearity(zero_filled_df, -0.80)

# Dropping columns
df_remain = zero_filled_df.drop(columns = cols_to_remove, axis = 1)

# Normalize using min_max function
df_normalized, tbut_norm = min_max(df_remain, tbut_values)

# Renaming columns
protein_names = list(df_remain.columns)
protein_names = pd.Series(protein_names)
df_normalized = df_normalized.rename(columns = protein_names)

"""""""""""""""
Q9: Create svm.SVC model with linear activation function
"""""""""""""""
X = df_normalized
Y = tbut_values

my_svm = svm.SVC(kernel = "linear").fit(X, Y)

"""""""""""""""
Q10: Make predictions for entire dataset
"""""""""""""""
my_prediction = my_svm.predict(X)

"""""""""""""""
Q11: Create confusion matrix between true values of TBUT & predicted values
"""""""""""""""
lin_conf_matrix = confusion_matrix(Y, my_prediction)
print(lin_conf_matrix)

"""""""""""""""
Q12: Compute accuracy, precision, and recall
"""""""""""""""
lin_accuracy = accuracy_score(Y, my_prediction)
print('Linear Kernel Accuracy Rate: {}%'.format(round(lin_accuracy * 100, 1)))

lin_precision, lin_recall = data_precision_recall(lin_conf_matrix)
print('Linear Kernel Precision: {}%'.format(round(lin_precision * 100, 1)))
print('Linear Kernel Recall: {}%'.format(round(lin_recall * 100, 1)))

"""""""""""""""
Q13: Repeat Q9-Q12 for svm.SVC model with 'rbf' activation function
"""""""""""""""
# Create model
X_rbf = df_normalized
Y_rbf = tbut_values

my_svm_rbf = svm.SVC(kernel = "rbf").fit(X_rbf, Y_rbf)

# Make predictions
my_prediction_rbf = my_svm_rbf.predict(X_rbf)

# Create confusion matrix
rbf_conf_matrix = confusion_matrix(Y_rbf, my_prediction_rbf)
print(rbf_conf_matrix)

# Compute accuracy, precision, and recall
rbf_accuracy = accuracy_score(Y_rbf, my_prediction_rbf)
print('RBF Kernel Accuracy Rate: {}%'.format(round(rbf_accuracy * 100, 1)))

rbf_precision, rbf_recall = data_precision_recall(rbf_conf_matrix)
print('RBF Kernel Precision: {}%'.format(round(rbf_precision * 100, 1)))
print('RBF Kernel Recall: {}%'.format(round(rbf_recall * 100, 1)))
