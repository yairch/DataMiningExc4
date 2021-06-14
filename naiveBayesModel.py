from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import pandas as pd
from os.path import join as path_join

M_ESTIMATOR = 2
pd.set_option('display.max_columns', None)


def import_data(directory_path, filename):
    full_path = path_join(directory_path, filename)
    try:
        resource = pd.read_csv(full_path)

        if resource is not None and isinstance(resource, pd.DataFrame) \
                and not (resource.empty or len(resource.columns) == 0):
            return resource
        else:
            return False
    except pd.errors.EmptyDataError:
        return False


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


def read_structure(directory_path, filename):
    """
    :param directory_path: path for the file to be read
    :param filename: structure file filename in directory
    :return: feature_structure: dict of attributes and their possible values
    """
    feature_structure = {}
    full_path = path_join(directory_path, filename)
    with open(full_path, 'r') as structure_file:
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


def refactor_prediction_labels(predictions, positive=1):
    """
    Refactor prediction labels for class attribute. yes = 1; no = 0.
    :param predictions: raw predictions
    :type predictions: list
    :param positive: value considered positive to be labeled as yes. Default is 1
    :type positive: int
    :return: relabeled predictions
    :rtype: list
    """

    return ['yes' if prediction == positive else 'no' for prediction in predictions]


def save_predictions(path, predictions, positive=1.0):
    """
    Writes predictions to output file "output.txt" in given path directory enumerated from 1.
    Predictions saved in format: yes = 1; no = 0.
    :param path: path to working directory
    :type path: string
    :param predictions: result classified predictions
    :type predictions: iterable
    :param positive: value considered positive to be labeled as yes. Default is 1
    :type positive: int
    """

    refactored_predictions = refactor_prediction_labels(predictions.tolist(), positive=positive)
    with open(path_join(path, 'output.txt'), 'w') as output:
        outputs = [str(i) + ' ' + str(prediction) for i, prediction in enumerate(refactored_predictions, 1)]
        output.write('\n'.join(outputs))


def build_model(train_set, feature_structure, n_bins=10):
    """
    Build and train a Naive Bayes Multinomial model with m-estimator = 2 according to given parameters.
    :param train_set: dataset to train on
    :type train_set: pandas.DataFrame
    :param feature_structure: dict of format: {feature: [values]} according to data structure file
    :type feature_structure: dict
    :param n_bins: number of bins for equal width discretization
    :type n_bins: int
    :return: trained model
    :rtype: sklearn.naive_bayes.MultinomialNB
    """

    fill_missing_values(train_set, feature_structure)
    discretize(n_bins=n_bins, dataset=train_set, feature_structure=feature_structure)
    model = MultinomialNB(alpha=M_ESTIMATOR)
    return train_naive_bayes_model(model=model, dataset=train_set, feature_structure=feature_structure)


def classify(model, test_set, feature_structure, n_bins=10):
    """
    Preprocess test set and make predictions for the test set
    :param model: trained naive bayes model
    :type model: sklearn.naive_bayes.MultinomialNB
    :param test_set: dataset to make predictions on
    :type test_set: pd.DataFrame
    :param feature_structure: dict of format: {feature: [values]} according to data structure file
    :type feature_structure: dict
    :param n_bins: number of bins for equal width discretization
    :type n_bins: int
    :return: classified predictions
    :rtype: numpy.ndarray
    """
    fill_missing_values(test_set, feature_structure)
    discretize(n_bins=n_bins, dataset=test_set, feature_structure=feature_structure)
    return classify_with_naive_bayes(classification_model=model, testset=test_set, feature_structure=feature_structure)
