# In this script, we will simulate MZ and DZ covariances from various level of A, C and E through multivariate normal distribution. Then we will use ACE model to estimate the A, C and E components and compare them with the true values.
# We will use the covairance matrices and the A,C,E values as training data to train a simple neural network
# to predict the A,C,E values from the covariance matrices. 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# function to simulate MZ and DZ covariances from A, C and E
def simulate_covariances(A, C, E, N_pairs):
    # MZ covariance matrix
    mz_cov = np.array([[A + C + E, A + C],
                       [A + C, A + C + E]])
    
    # DZ covariance matrix
    dz_cov = np.array([[A + C + E, 0.5 * A + C],
                       [0.5 * A + C, A + C + E]])
    
    # Simulate MZ and DZ data
    df_mz = np.random.multivariate_normal(mean=[0, 0], cov=mz_cov, size=N_pairs)
    df_dz = np.random.multivariate_normal(mean=[0, 0], cov=dz_cov, size=N_pairs)
    
    # get the covariance matrices from the simulated data
    mz_cov_sim = np.cov(df_mz, rowvar=False)
    dz_cov_sim = np.cov(df_dz, rowvar=False)
    return mz_cov_sim, dz_cov_sim


    
    