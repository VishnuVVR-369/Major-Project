# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Data
train_data = pd.read_csv('../DATA/data.csv')

# %%
train_data.head()

# %%
print("NUMBER OF DATA POINTS -", train_data.shape[0])
print("NUMBER OF FEATURES -", train_data.shape[1])
print("FEATURES -", train_data.columns.values)

# %%
# Data Description using describe() in pandas
train_data.describe().style.set_properties(
    **{'background-color': '#F0F0F0', 'color': '#222222', 'border': '1.5px  solid black'})

# %%
# Data Description using info() in pandas
train_data.info()

# %%
