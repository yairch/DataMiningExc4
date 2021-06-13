from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import pandas as pd

M_ESTIMATOR = 2

model = MultinomialNB(alpha=M_ESTIMATOR)

pd.set_option('display.max_columns', None)


def extract_feature(line, feature_structure):
    """
    :param line: single line as read from the train.csv file
    :param feature_structure: dict of attributes and their possible values
    :return: -
    """
    tokens = line.split()
    feature = tokens[1]
    raw_values = ' '.join(tokens[2:])
    values = [value for value in raw_values.strip('{}').split(',')]

    feature_structure[feature] = values


def read_structure(path):
    """
    :param path: path for the file to be read
    :return: feature_structure: dict of attributes and their possible values
    """
    feature_structure = {}
    with open(path, 'r') as structure_file:
        for line in structure_file.readlines():
            extract_feature(line, feature_structure)

    return feature_structure


def fill_missing_values(dataset, feature_structure):
    """
    :param dataset: data frame containing missing values
    :param feature_structure: dict of attributes and their possible values
    """
    for feature in dataset:
        if feature == 'class':
            continue

        if feature_structure[feature][0] == 'NUMERIC':
            dataset[feature].fillna(dataset[feature].mean(), inplace=True)
        else:
            dataset[feature].fillna(dataset[feature].mode(), inplace=True)


def discretize(n_bins, dataset, feature_structure):
    """
    :param n_bins: number of bins to perform equal width discretization on.
    :param dataset: data frame for discretization
    :param feature_structure: dict of attributes and their possible values
    """
    equal_width_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    for feature in dataset:
        if feature_structure[feature][0] == 'NUMERIC':
            discrete_feature = equal_width_discretizer.fit_transform(dataset[[feature]]).reshape(-1)
            dataset[feature] = pd.Series(discrete_feature)


def to_numerical(dataset, feature_structure):
    """
    converting categorical variables into numeric variables using label encoder
    :param dataset: the data frame
    :param feature_structure: dict of attributes and their possible values
    :return:
    """
    le = LabelEncoder()
    for feature in feature_structure.keys():
        dataset[feature] = le.fit_transform(dataset[feature])
    return dataset


def train_naive_bayes_model(model, dataset, feature_structure):
    """
    :param model: naive bayes model
    :param dataset: the training set data frame
    :param feature_structure: dict of attributes and their possible values
    :return:
    """
    dataset = to_numerical(dataset, feature_structure)
    X_train, y_train = dataset.drop('class', axis=1), dataset['class']
    return model.fit(X_train, y_train)


def classify_with_naive_bayes(classification_model, testset, feature_structure):
    """
    :param classification_model: naive bayes model
    :param testset: the dataset containing data for classification
    :param feature_structure: dict of attributes and their possible values
    :return: model's prediction
    """
    testset = to_numerical(testset, feature_structure)
    predictores = list(feature_structure.keys())
    predictores.remove('class')
    res = classification_model.predict(testset[predictores])
    return res

####### add function to write results into file##########

###### call the training function in GUI.py and send all parameters with it #########

##### call funciton for classify ################

##### IN GUI - add message after successful classification ###############


df = pd.read_csv("D:\\G I L A\\data science\\211702782_206085532\\DataMiningExc4\\train.csv")
#
struct = read_structure("D:\\G I L A\\data science\\211702782_206085532\\DataMiningExc4\\Structure.txt")
print("Struct:")
print(struct)
# print("with missing:")
# print(df.head())
print("no missing:")
fill_missing_values(df, struct)
print(df.head())
print("discrete:")
discretize(n_bins=10, dataset=df, feature_structure=struct)
print(df.head())
trained_model = train_naive_bayes_model(model, df, struct)

### test model ###

test_df = pd.read_csv("D:\\G I L A\\data science\\211702782_206085532\\DataMiningExc4\\test.csv")
fill_missing_values(test_df, struct)
discretize(n_bins=10, dataset=test_df, feature_structure=struct)
classify_with_naive_bayes(trained_model, test_df, struct)
