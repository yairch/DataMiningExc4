from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

M_ESTIMATOR = 2

model = CategoricalNB(alpha=M_ESTIMATOR)

pd.set_option('display.max_columns', None)

def extract_feature(line, feature_structure):
    tokens = line.split()
    feature = tokens[1]
    raw_values = ' '.join(tokens[2:])
    values = [value for value in raw_values.strip('{}').split(',')]

    feature_structure[feature] = values


def read_structure(path):
    feature_structure = {}
    with open(path, 'r') as structure_file:
        for line in structure_file.readlines():
            extract_feature(line, feature_structure)

    return feature_structure


def fill_missing_values(dataset, feature_structure):
    for feature in dataset:
        if feature == 'class':
            continue

        if feature_structure[feature][0] == 'NUMERIC':
            dataset[feature].fillna(dataset[feature].mean(), inplace=True)
        else:
            dataset[feature].fillna(dataset[feature].mode(), inplace=True)


def discretize(n_bins, dataset, feature_structure):
    equal_width_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    for feature in dataset:
        if feature_structure[feature][0] == 'NUMERIC':
            discrete_feature = equal_width_discretizer.fit_transform(dataset[[feature]]).reshape(-1)
            dataset[feature] = pd.Series(discrete_feature)


# df = pd.read_csv("D:\Programming\BGU\Data Mining and Data Warehousing\Exc4\\train.csv")
#
# struct = read_structure("D:\Programming\BGU\Data Mining and Data Warehousing\Exc4\Structure.txt")
# print("Struct:")
# print(struct)
# print("with missing:")
# print(df.head())
# print("no missing:")
# fill_missing_values(df, struct)
# print(df.head())
# print("discrete:")
# discretize(n_bins=10, dataset=df, feature_structure=struct)
# print(df.head())
