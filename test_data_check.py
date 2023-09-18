# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:00:38 2023

@author: ADAMS-LAB
"""
import pickle
import numpy as np
filename = 'Results/MRTA_Flood/MRTA_Flood_nloc_101_nrob_6_ND_CAPAM_K_2_P_2_Le_2_h_128'

# Open the file in binary read mode
with open(filename, 'rb') as file:
    # Load the object from the file
    obj = pickle.load(file)

# obj now contains the loaded object
print(np.mean(obj['total_tasks_done']))