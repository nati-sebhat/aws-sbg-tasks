import numpy as np
import pandas as pd

df = pd.read_csv('week-01/data/student-mat.csv', sep=";")

print(df.head())

print(df.shape)

print(df.columns)

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.dtypes)

print("Max Values")
print(df[['G1', 'G2', 'G3']].max())

print("Min values")
print(df[['G1', 'G2', 'G3']].min())



def check_missing_values(df):
    return df.isnull().sum()

print(check_missing_values(df))


def check_duplicates(df):
    return df.duplicated().sum()

print(check_duplicates(df))


def clean_missing_values(df):
    return df.dropna()

print(clean_missing_values(df))    


cleaned_df = clean_missing_values(df)
print(cleaned_df.shape)


print(check_missing_values(df).sum())
print(check_duplicates(df))
print(cleaned_df.shape)



print("Average:", np.mean(df['G3']))
print("Highest:", np.max(df['G3']))
print("Lowest:", np.min(df['G3']))


def calculate_grade_stats(df):
    grade = df['G3']
    average = np.mean(grade)
    highest = np.max(grade)
    lowest = np.min(grade)
    return average, highest, lowest
print(calculate_grade_stats(df))    



stats = calculate_grade_stats(df)

print(stats[0])
print(stats[1])
print(stats[2])

above = df['G3'] >= 15
print(above.sum())



count = above.sum()
total = len(df['G3'])
percentage = count / total * 100
print(percentage)


def calculate_grade_percentage(df, threshold):
    number = df['G3'] >= threshold
    count = number.sum()
    total = len(df['G3'])
    per = count / total * 100
    return per
print(calculate_grade_percentage(df, 15))    
print(calculate_grade_percentage(df, 10))




print("Mean value for G1:", np.mean(df['G1']))
print("Mean value for G2:", np.mean(df['G2']))
print("Mean value for G3:", np.mean(df['G3']))


def calculate_mean(df, column):
    mean_value = np.mean(df[column])
    return mean_value

print(calculate_mean(df, 'G1'))
print(calculate_mean(df, 'G2'))
print(calculate_mean(df, 'G3'))


g1_value = np.mean(df['G1'])
g3_value = np.mean(df['G3'])
subtract = g1_value - g3_value

print("Difference between G1 and G3 is:", subtract)
