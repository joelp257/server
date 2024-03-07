# -*- coding: utf-8 -*-
"""regressionglucose.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_IUn12095wJ8GflohOfu8hVr0dXaCS19
"""

import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv("diabetes.csv")

# Explore the data
print("Shape of the dataset:", data.shape)
print("First 5 rows of the dataset:\n", data.head())

X= data.iloc[:,:-1]

# prompt: import the pycaret regresssion



from pycaret.regression import *

import pycaret
reg = pycaret.regression.setup(data=X, target="Glucose")

reg.compare_models()

best_model = reg.compare_models()

evaluate_model(best_model)

with open('regglucose.pickle', 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)