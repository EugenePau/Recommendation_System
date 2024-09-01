from src.data_preprocessing import DF_generate, DF_enhanced
from src.utils import plot_rating_distribution, count_unique_values, check_DF, check_top_reviewers
from src.model import svd_model, cf_recommendation

def main():
    ## INPUT HERE ##
    datafile = 'dataset.csv'
    columns = ['userId', 'productId','Rating','timestamp']
    irrelevance = ['timestamp']
    customer_id = 432
    ## END OF INPUT ##

    # Data preprocessing (Input)
    df = DF_generate(datafile,columns,irrelevance)

    # Data preview
    check_DF(df)
    count_unique_values(df)
    check_top_reviewers(df)

    # Statistical Plot 
    plot_rating_distribution(df)

    # Propagate model
    svd = svd_model(df)

    # Recommendation
    cf_recommendation(df,svd,customer_id)

    print('End of Program')
    return


if __name__ == "__main__" :
    main()
