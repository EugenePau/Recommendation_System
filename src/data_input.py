import numpy as np # linear algebra
import pandas as pd # main Data Structure for processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns # statistical visualisation



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Import data frame
electronics_data = pd.read_csv("dataset.csv",names=['userId', 'productId','Rating','timestamp'])
#Dropping the Timestamp column (column name, axis to remove (0:row, 1:column), if rewrite or save as)
electronics_data.drop(['timestamp'], axis=1,inplace=True)
# Data preview (first 10 enetries)
print('DataFrame Preview: (First 10 entries)')
print(electronics_data.head(10))
# Dimension of Matrix (row,column)
print(f"Table Dimension: {electronics_data.shape}")
#Check for missing values
print('Number of missing values across columns: \n',electronics_data.isnull().sum())


print("-"*50)

#Statistical analysis: mean, standard deviation, quartiles
print('Statistics for rating distribution:')
print(electronics_data.describe()['Rating'])
# Visualise the distribution of the rating, through context manager block
print('Close the window to continue...')
with sns.axes_style('white'):
    g = sns.catplot(x="Rating", data=electronics_data, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
    plt.show()

# Unique Count
print("-"*50)
print("\nTotal no of ratings :",electronics_data.shape[0])
print("Total No of Users   :", len(np.unique(electronics_data.userId)))
print("Total No of products  :", len(np.unique(electronics_data.productId)))
print('\n')

# Analysis of rating given by users 
no_of_rated_products_per_user = electronics_data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
print('Top 5 reviewers:')
print(no_of_rated_products_per_user.head(5))
print('Statistics for No. of ratings given per person:')
print(no_of_rated_products_per_user.describe())

print("-"*50)

#Getting the new dataframe which contains users who has given 50 or more ratings
new_df=electronics_data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)

## model part ; caution NumPy must be <2.0
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import SVD
import os
from surprise.model_selection import train_test_split

#Reading the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df,reader)

#Splitting the dataset
trainset, testset = train_test_split(data, test_size=0.3,random_state=10)

# Instantiate the SVD model
svd = SVD()

# Train the model on the training set
svd.fit(trainset)

# Run the trained model against the test set
predict = svd.test(testset)

# Compute and print RMSE and MAE
rmse = accuracy.rmse(predict)
mae = accuracy.mae(predict)

## Recommendation System

# Capture all predicted ratings of a user
user_id = str(345)
predictions = []
all_item_ids = new_df['productId'].unique()
for item_id in all_item_ids:
    predicted_rating = svd.predict(user_id,item_id).est
    predictions.append((item_id,predicted_rating))

# Create a new DataFrame to host all prediction of a user
predictions_df = pd.DataFrame(predictions,columns=['itemid','predicted_rating'])

# Sort and Select Top 6 best matches
top_recommendations = predictions_df.sort_values(by='predicted_rating',ascending=False)
print(top_recommendations.head(6))

print('end')
print("-"*50)
