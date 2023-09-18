# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:07:15 2023

@author: ADAMS-LAB
"""

import os
import os
import numpy as np

# Specify the directory
dir_path = 'action_val'

# List to store the standard deviations
std_devs = []

# Get a list of all .npy files in the directory along with their timestamps
npy_files_with_time = [(file_name, os.path.getmtime(os.path.join(dir_path, file_name)))
                       for file_name in os.listdir(dir_path) if file_name.endswith('.npy')]

# Sort the files by their modification time
sorted_npy_files = [file_name for file_name, _ in sorted(npy_files_with_time, key=lambda x: x[1])]

# Iterate through sorted files in the directory
for file_name in sorted_npy_files:
    file_path = os.path.join(dir_path, file_name)

    # Load the numpy array from the file
    loaded_array = np.load(file_path)

    # Calculate the standard deviation of the array
    std_dev = np.std(loaded_array) # If you want mean, use np.mean

    # Append the standard deviation to the list
    std_devs.append(std_dev)

# Now std_devs contains the standard deviations of the values in the .npy files, in order of modification time
