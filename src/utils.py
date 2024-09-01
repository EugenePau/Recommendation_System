import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_rating_distribution(df):
    #Statistical analysis: mean, standard deviation, quartiles
    print('Statistics for rating distribution:')
    print(df.describe()['Rating'])
    # Visualise the distribution of the rating, through context manager block
    print('Close the window to continue...')
    with sns.axes_style('white'):
        g = sns.catplot(x="Rating", data=df, aspect=2.0,kind='count')
        g.set_ylabels("Total number of ratings")
        plt.show()
    print('-'*50)
    return

def check_DF(df):
    # DataFrame preview (first 10 enetries)
    print('DataFrame Preview: (First 10 entries)')
    print(df.head(10))
    print('-'*50)
    return

def count_unique_values(df):
    print("Total no of ratings :",df.shape[0])
    print("Total No of Users   :", len(np.unique(df.userId)))
    print("Total No of products  :", len(np.unique(df.productId)))
    print('-'*50)
    return


def check_top_reviewers(df):
    no_of_rated_products_per_user = df.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
    print('Top 5 reviewers:')
    print(no_of_rated_products_per_user.head(5))
    print('Statistics for No. of ratings given per person:')
    print(no_of_rated_products_per_user.describe())
    print('-'*50)
    return


