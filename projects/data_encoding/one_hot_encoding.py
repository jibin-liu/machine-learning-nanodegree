from sklearn import preprocessing
import pandas as pd
import numpy as np


# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
               'health': ['fit', 'slim', 'obese', 'fit', 'slim'],
               'height': ['high', 'middle', 'middle', 'short', 'high']}

# store into data frame
df = pd.DataFrame(sample_data, columns=['name', 'health', 'height'])

# In pandas - One Hot encoding
one_hot_pandas = pd.get_dummies(df[['health', 'height']])
print(one_hot_pandas)

# In sklearn
# convert text to numerical labels
label_encoder = preprocessing.LabelEncoder()
health = label_encoder.fit_transform(df['health'])
height = label_encoder.fit_transform(df['height'])
labels = np.array([health, height]).T

onehot_encoder = preprocessing.OneHotEncoder()
# rows are samples, and columns are features.
one_hot_sklearn = onehot_encoder.fit_transform(labels)
print(one_hot_sklearn)
