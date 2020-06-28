#import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read the CSV file into a dataframe
diabetes = pd.read_csv('diabetes.csv')

#let see the first 5 records
diabetes.head()
diabetes.columns
diabetes.info()
diabetes.describe()
#lets check for Null vaules.b
diabetes.isnull().sum()

######## from INFO and ISNULL check it is clear that data has no NULL values #######
#lets check the correlation between each feature
diabetes.corr()

%matplotlib inline
plt.figure(figsize=(10,8))
sns.heatmap(diabetes.corr(),annot=True)

#####   from corelation matrix we could see Glocose has the highest relation with the Target variable.  ####
#lets do a pair plot
plt.figure(figsize=(12,10))
sns.pairplot(diabetes)

sns.jointplot('Glucose','Outcome',data=diabetes)
sns.distplot(diabetes['Glucose'])

## So from all the plot it seems that higher the value of Glucose, more is the chance of Diabetics which is obvious in a real world.
