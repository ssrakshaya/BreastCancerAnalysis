# =====================
# 📦 Imports
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo


# =====================
# 📥 Load Dataset
# =====================
# Fetch the UCI Breast Cancer Wisconsin dataset
breast_cancer = fetch_ucirepo(id=17)

# Full original data (for plotting, etc.)
df = breast_cancer.data.original

# Split into features and target
X = breast_cancer.data.features
y = breast_cancer.data.targets


# =====================
# 🧠 Explore Metadata
# =====================
print("=== Dataset Metadata ===")
print(breast_cancer.metadata)

print("\n=== Variable Info ===")
print(breast_cancer.variables)


# =====================
# 🔍 Quick Look at Data
# =====================
print("\n=== Feature Sample ===")
print(X.head())

print("\n=== Target Sample ===")
print(y.head())

print("\n=== Diagnosis Labels ===")
print(y['Diagnosis'].unique())


#creating smaller subset of data frame seperated by target class (malignant vs benign)
#benign tumors
benign = df[df["Diagnosis"] == 'B']
#malignant tumors
malignant= df[df["Diagnosis"] == 'M']


#uSing the columns of uniformity of cell shape vs uniformity of cell size from the subset dataframes
plt.scatter(benign['Uniformity_of_cell_shape'], benign['Uniformity_of_cell_size'], color='red', marker='o', label='Benign Tumor')
plt.scatter(malignant['Uniformity_of_cell_shape'], malignant['Uniformity_of_cell_size'], color='blue', marker='x', label = 'Malignant Tumor')

plt.xlabel("Cell Shape Uniformity")
plt.ylabel("Cell Size Uniformity")
plt.legend(loc='upper left')

plt.show()