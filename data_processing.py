from pandas import read_csv, DataFrame, concat
import numpy as np

# Remove id column and replace missing values for the mean value of the attribute
cancer = read_csv('tests/database/breast-cancer-wisconsin.data', index_col=0, header=None, na_values='?')
rows, cols = np.where(cancer.isnull() == True)
cancer = cancer.values

for j in set(cols):
    cancer[rows, j] = np.nanmean(cancer[:, j])

DataFrame(cancer).to_csv('tests/database/breast-cancer-wisconsin-processed.csv', header=False, index=False)

# Encode labels and remove zeroed feature
iono = read_csv('tests/database/ionosphere.data', index_col=None, header=None).values
iono = np.delete(iono, 1, 1)
classes = np.unique(iono[:, -1])

for k, c in enumerate(classes):
    i = np.where(iono[:, -1] == c)
    iono[i, -1] = k

DataFrame(iono).to_csv('tests/database/ionosphere-processed.csv', header=False, index=False)

# Convert to csv and remove header
pima = read_csv('tests/database/pima.tsv', index_col=None, header=0, sep='\t')
pima.to_csv('tests/database/pima-processed.csv', header=False, index=False, sep=',')

# Put target column as last column
wine = read_csv('tests/database/wine.data', index_col=None, header=None)
x, y = wine.iloc[:, 1:], wine.iloc[:, 0]

concat([x, y], axis=1).to_csv('tests/database/wine-processed.csv', header=False, index=False)