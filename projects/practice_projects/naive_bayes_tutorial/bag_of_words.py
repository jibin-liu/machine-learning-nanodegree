""" implement bag of words from scratch """
import string
from collections import Counter


def scratch_implementation():
    documents = [
        'Hello, how are you!',
        'Win money, win from home.',
        'Call me now.',
        'Hello, Call hello you tomorrow?'
    ]

    # step 1: convert all strings to their lower case form
    lower_case_documents = (s.lower() for s in documents)

    # Step 2: Removing all punctuations
    sans_punctuation_documents = (s.translate(str.maketrans('', '', string.punctuation))
                                  for s in lower_case_documents)

    # Step 3: Tokenization
    preprocessed_documents = (s.split(' ') for s in sans_punctuation_documents)

    # Step 4: Count frequencies
    frequency_list = (Counter(words) for words in preprocessed_documents)
    print(next(frequency_list))


def scikit_learn_implementation():
    """ use CountVectorizer to implement bag of words """
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    # initialize counter
    count_vector = CountVectorizer()

    documents = [
        'Hello, how are you!',
        'Win money, win from home.',
        'Call me now.',
        'Hello, Call hello you tomorrow?'
    ]

    # fit the documents into countVectorizer
    count_vector.fit(documents)

    # turn vector into a matrix, in which each row corresponds to each document,
    # and each column for each words. The value at (row, col) is the frequency
    # of the word
    doc_array = count_vector.transform(documents).toarray()

    # turn array into a dataframe, with column names
    frequency_matrix = pd.DataFrame(doc_array,
                                    columns=count_vector.get_feature_names())
    print(frequency_matrix)


if __name__ == '__main__':
    scratch_implementation()
    scikit_learn_implementation()
