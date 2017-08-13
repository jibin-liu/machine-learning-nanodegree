import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def read_input(input_path):
    """
    1. Import the dataset into a pandas dataframe using the read_table method.
    Because this is a tab separated dataset we will be using '\t' as the value
    for the 'sep' argument which specifies this format.
    2. Also, rename the column names by specifying a list
    ['label, 'sms_message'] to the 'names' argument of read_table().
    3. Print the first five values of the dataframe with the new column names.
    """
    df = pd.read_table(input_path, sep=r'\t', names=['label', 'sms_message'],
                       header=None)
    return df


def converting_label_to_numeric(df):
    """
    1. Convert the values in the 'label' colum to numerical values using map
    method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0 and
    the 'spam' value to 1.
    2. Also, to get an idea of the size of the dataset we are dealing with,
    print out number of rows and columns using 'shape'.
    """
    df.label = df.label.map({'ham': 0, 'spam': 1})
    return df


def split_data(df):
    """ Split the dataset into a training and testing set by using the
    train_test_split method in sklearn."""
    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                        df['label'],
                                                        test_size=0.33,
                                                        random_state=47)
    print(f'total set: {df.shape[0]}')
    print(f'training set: {X_train.shape[0]}')
    print(f'test set: {X_test.shape[0]}')
    return X_train, X_test, y_train, y_test


def applying_bag_of_words(X_train, X_test):
    """ applying bag of words on training and test input data """
    vecterizor = CountVectorizer()
    vecterizor.fit(X_train)
    train_matrix = vecterizor.transform(X_train).toarray()
    test_matrix = vecterizor.transform(X_test).toarray()
    return train_matrix, test_matrix


def training(X_training, y_train):
    """ training data using MultinomialNB """
    model = MultinomialNB()
    model.fit(X_training, y_train)
    return model


def predict(model, X_testing):
    """ predict using trained model """
    return model.predict(X_testing)


def evaluate(predictions, y_test):
    """
    compute the accuracy, precision, recall, and F1 scores of the model
    """
    # accuracy = number of correct predictions / all predictions
    num_correct_predictions = sum(predictions == y_test)
    accuracy = num_correct_predictions / y_test.shape[0]
    print(f'accuracy = {accuracy}')

    # precision = true positive / all positives
    all_positive_idx = (predictions == 1)
    true_pos = np.sum(predictions[all_positive_idx] == y_test[all_positive_idx])
    precision = true_pos / y_test[all_positive_idx].shape[0]
    print(f'precision = {precision}')

    # recall = true positive / (true positive + false negative)
    all_negative_idx = (predictions == 0)
    false_neg = np.sum(predictions[all_negative_idx] != y_test[all_negative_idx])
    recall = true_pos / (true_pos + false_neg)
    print(f'recall = {recall}')

    # F1 = weighted average of the precision and recall scores
    f1 = (precision + recall) / 2
    print(f'F1 score = {f1}')

    # comparing to sklearn functions
    print('\n')
    print(f'sklearn accuracy = {accuracy_score(y_test, predictions)}')
    print(f'sklearn precision = {precision_score(y_test, predictions)}')
    print(f'sklearn recall = {recall_score(y_test, predictions)}')
    print(f'sklearn F1 score = {f1_score(y_test, predictions)}')


def main():
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    input_path = './smsspamcollection/SMSSpamCollection'

    # read input
    df = read_input(input_path)

    # turn label into numeric values
    df = converting_label_to_numeric(df)

    # split data into training and test set
    X_train, X_test, y_train, y_test = split_data(df)

    # applying bag of words on input data
    training_data, testing_data = applying_bag_of_words(X_train, X_test)

    # fit training data using MultinomialNB
    naive_bayes = training(training_data, y_train)

    # predict using testing data
    predictions = predict(naive_bayes, testing_data)

    # evaluating the model
    evaluate(predictions, y_test)

if __name__ == '__main__':
    main()
