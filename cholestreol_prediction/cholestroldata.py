# -*- coding: utf-8 -*-
"""cholestroldata.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1up9IXHgwj2hd2L4XT9pjIUc8mkfOJraT
"""

import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv("dataset_2190_cholesterol.csv")

# Explore the data
print("Shape of the dataset:", data.shape)
print("First 5 rows of the dataset:\n", data.head())

X= data.iloc[:,:]

# prompt: import the pycaret regresssion



from pycaret.regression import *

import pycaret
reg = pycaret.regression.setup(data=X, target="chol")

best_model = reg.compare_models()
col_names_list=data.columns.tolist()
pd.Series(col_names_list).to_pickle('chol_pred_col_names.pkl')

evaluate_model(best_model)

save_model(best_model,"chol_pred")