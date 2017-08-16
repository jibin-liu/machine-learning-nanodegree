from sklearn import preprocessing
import pandas as pd


# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
               'health': ['fit', 'slim', 'obese', 'fit', 'slim']}

# store into data frame
df = pd.DataFrame(sample_data, columns=['name', 'health'])

# user LabelEncoder to transform data to numerical
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df['health'])
new_health = label_encoder.transform(df['health'])
print(new_health)


# the above is same as
new_health_2 = label_encoder.fit_transform(df['health'])
print(new_health_2)
