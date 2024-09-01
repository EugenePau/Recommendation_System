import pandas as pd
import os

def DF_generate(filename,columns,irrelevance): ## Output DF contains: 'userId', 'productId','Rating'
    data_path = data_path = data_path = os.path.join('data',filename)
    df = pd.read_csv(data_path,names=columns)
    #Dropping irrelevant columns
    df.drop(irrelevance, axis=1,inplace=True)
    # Dimension of Matrix (row,column)
    print(f"Table Dimension: {df.shape}")
    #Check for missing values
    print('\nNumber of missing values across columns: \n',df.isnull().sum())
    return df



def DF_enhanced(df):
    # Improved dataframe which only contains users who has given >50 ratings
    new_df = df.groupby("productId").filter(lambda x:x['Rating'].count() >=50)
    return df
