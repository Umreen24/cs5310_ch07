# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""""""""""""""
IMPORTING PACKAGES
"""""""""""""""
import os
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from math import log2, ceil

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
    
    schirmer_scaled = minmax_scale(array)
    
    return normalied_corr_df, schirmer_scaled

# Function to create line plot
def create_lineplot(data, data2, col_nums, figure_name):
    sns.set_theme(style = 'white')
    
    plt.plot(data, label = 'Schirmer Collection Values', color = 'red')
    plt.plot(data2.iloc[:,:].mean(axis = 1), label = 'Avg. of Other Proteins', color = 'blue')
    plt.legend(loc = 'upper right')
    plt.title('Line Plot of Schirmer Collection & Proteins', fontsize = 18)
    plt.xlabel('Row', fontsize = 12)
    plt.ylabel('Value', fontsize = 12)
    plt.savefig(figure_name)
    plt.show
    
"""""""""""""""
Q1-Q2: Load data into dataframe and extract "Schirmer collection"
"""""""""""""""
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/chapter07/lab')
filename = 'PatientInfo.xlsx'
data_df = load_read_data(filename)

# Extract "Schirmer collection" column
# Use .values property to extract column into array
schirmer = data_df.iloc[:, 5].values

"""""""""""""""
Q3: Load protein data into dataframe
"""""""""""""""
protein_filename = 'Data_SubjectStripTearSamples-Dec2020_Report(1).xlsx'
protein_data_df = load_read_data(protein_filename)

"""""""""""""""
Q4: Remove unecessary columns
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
df_normalized, schirmer_norm = min_max(df_remain, schirmer)

# Renaming columns
protein_names = list(df_remain.columns)
protein_names = pd.Series(protein_names)
df_normalized = df_normalized.rename(columns = protein_names)

"""""""""""""""
Q9: Create line plot of "Schirmer collection" and avg. of experimental data
"""""""""""""""
cols = df_normalized.shape[1]
values_plot = create_lineplot(schirmer_norm, df_normalized, cols, 'fig1.png')

"""""""""""""""
Q10-Q11: Create KerasRegressor with two hidden layers and visualize model
"""""""""""""""
# Set the number of input variables
ZZZ = df_normalized.shape[1]

# Setup inputs for neural network model
# Find the power of 2 that is nearest to the number of input variables (ZZZ)

def power_of_2(a):
    
    if a == 0:
        return 1 
    else:
        return 2 ** ceil(log2(a))


x = power_of_2(ZZZ)
NN1 = round(x / 2)

def find_factors(n):
    factors = []
    
    for i in range(1, x + n):
        if n % i == 0:
            factors.append(i)
    
    return factors

factors = find_factors(NN1)
NN2 = factors[5]

# Setup the Keras model
def baseline_model():
    # Create model
    model = Sequential()
    # The term "Dense" means layer. When we add extra "Dense()" to the model
    # that means we are adding extra layers to the neural network model.
    model.add(Dense(NN1, input_dim = ZZZ, activation = 'relu'))
    model.add(Dense(NN2, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    model.summary()
    dot_img_file = 'model.png'
    tf.keras.utils.plot_model(model, to_file = dot_img_file, 
                              show_shapes = True)
    return model

estimator = KerasRegressor(build_fn = baseline_model, epochs = 10, 
                            batch_size = 256, verbose = 0)

"""""""""""""""
Q12: Make predictions for entire dataset
"""""""""""""""
estimator.fit(df_normalized, schirmer_norm)
predicted_values = estimator.predict(df_normalized)


"""""""""""""""
Q13: Compute correlation coefficient and mean absolute error (MAE)
"""""""""""""""
model_corr, _ = pearsonr(schirmer_norm, predicted_values)
model_mae = mean_absolute_error(schirmer_norm, predicted_values)

print('Pearsons correlation: %.3f' % model_corr)
print('Mean absolute error: %.3f' % model_mae)