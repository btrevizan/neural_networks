from pandas import read_csv, DataFrame, concat
import numpy as np


def normalize(data):
    for j in range(data.shape[1]):
        dmin = np.min(data[:, j])
        data[:, j] = (data[:, j] - dmin) / (np.max(data[:, j]) - dmin)

    return data

# Remove id column and replace missing values for the mean value of the attribute
cancer = read_csv('tests/datasets/breast-cancer-wisconsin.data', index_col=0, header=None, na_values='?')

rows, cols = np.where(cancer.isnull() == True)

cancer = cancer.values
classes = np.unique(cancer[:, -1])

for j in set(cols):
    for c in classes:
        r = np.where(cancer[:, -1] == c)
        cancer[rows, j] = np.nanmean(cancer[r, j])

DataFrame(cancer).to_csv('tests/datasets/breast-cancer-wisconsin-processed.csv', header=False, index=False)

# Encode labels and remove zeroed feature
iono = read_csv('tests/datasets/ionosphere.data', index_col=None, header=None).values
iono = np.delete(iono, 1, 1)
classes = np.unique(iono[:, -1])

for k, c in enumerate(classes):
    i = np.where(iono[:, -1] == c)
    iono[i, -1] = k

iono = normalize(iono)
DataFrame(iono).to_csv('tests/datasets/ionosphere-processed.csv', header=False, index=False)

# Convert to csv and remove header
pima = read_csv('tests/datasets/pima.tsv', index_col=None, header=0, sep='\t').values

for j in range(pima.shape[1]):
    dmax = np.max(pima[:, j])
    pima[:, j] = (pima[:, j] - dmax) / (dmax - np.min(pima[:, j]))

pima = normalize(pima)
DataFrame(pima).to_csv('tests/datasets/pima-processed.csv', header=False, index=False, sep=',')

# Put target column as last column
wine = read_csv('tests/datasets/wine.data', index_col=None, header=None)
x, y = wine.iloc[:, 1:], wine.iloc[:, 0]
wine = concat([x, y], axis=1).values

wine = normalize(wine)
DataFrame(wine).to_csv('tests/datasets/wine-processed.csv', header=False, index=False)
