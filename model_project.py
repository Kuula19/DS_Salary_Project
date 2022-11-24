# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:04:50 2022

@author: Fatima-ezzahra Badaoui

"""
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline

df = pd.read_csv("eda_data.csv")
df

print(os.getcwd())