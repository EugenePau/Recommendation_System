from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split

import pandas as pd
import os


def svd_model(df):
    print('Constructing Model...\n')

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df,reader)

    #Splitting the dataset
    trainset, testset = train_test_split(data, test_size=0.3,random_state=10)
    # Instantiate the SVD model
    svd = SVD()
    # Train the model on the training set
    svd.fit(trainset)
    # Run the trained model against the test set
    predict = svd.test(testset)

    # Evaluation
    print('Model Evaluation:')
    rmse = accuracy.rmse(predict)
    mae = accuracy.mae(predict)
    print('-'*50)
    return svd

def cf_recommendation(df,svd,uid):
    # Capture all predicted ratings of a user
    user_id = str(uid)
    predictions = []
    all_item_ids = df['productId'].unique()
    for item_id in all_item_ids:
        predicted_rating = svd.predict(user_id,item_id).est
        predictions.append((item_id,predicted_rating))

    # Create a new DataFrame to host all prediction of a user
    predictions_df = pd.DataFrame(predictions,columns=['itemid','predicted_rating'])

    # Sort and Select Top 6 best matches
    top_recommendations = predictions_df.sort_values(by='predicted_rating',ascending=False)
    print('Top 6 recommendations:')
    print(top_recommendations.head(6))
    print('-'*50)
    return top_recommendations


