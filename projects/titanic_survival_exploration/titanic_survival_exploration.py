import pandas as pd
import numpy as np


def load_data(csv_path):
    """ load csv into data frame """
    df = pd.read_csv(csv_path)
    return df


def split_outcomes(df):
    """ split survivals results out of df """
    outcomes = df['Survived']
    df = df.drop('Survived', axis=1)
    return outcomes, df


def accuracy_score(y_test, predictions):
    """ calculate the accuracy score of the prediction """
    # my implementation
    # correct_prediction = y_test == predictions
    # accuracy_score = (predictions[correct_prediction].shape[0] /
    #                   (predictions.shape[0] * 1.0))

    # better implementation
    if len(y_test) != len(predictions):
        raise ValueError('y_test and predictions are in different shape')

    return (y_test == predictions).mean()


def vis_survival_stats(data, outcomes, feature):
    """ visualize the data by category """
    pass


def main():
    # load csv into data frame
    csv_path = './titanic_data.csv'
    df = load_data(csv_path)
    print(df.dtypes)

    # split outcomes and features
    outcomes, features = split_outcomes(df)

    # evaluate model by calculate accuracy score
    accuracy = accuracy_score(outcomes, np.zeros_like(outcomes))
    print(accuracy)




if __name__ == '__main__':
    main()
