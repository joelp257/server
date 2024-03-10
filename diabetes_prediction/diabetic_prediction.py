

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

X= data.iloc[:,:]

# prompt: import the pycaret regresssion



from pycaret.classification import *

import pycaret
clf = pycaret.classification.setup(data=X, target="Outcome")

clf.compare_models()

best_model = clf.compare_models()

#evaluate_model(best_model)

col_names_list=data.columns.tolist()
pd.Series(col_names_list).to_pickle('d_pred_col_names.pkl')

save_model(best_model,"d_pred")