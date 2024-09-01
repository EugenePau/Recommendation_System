"# Recommendation_System" 
# Amazon Review Recommendation System

## Overview

The Amazon Review Recommendation System is a  project designed to provide personalized recommendations based on user reviews. The system analyzes product reviews and generates suggestions using Collaborative Filtering and Matrix Factorization (SVD) techniques.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Features

- Analyze and preprocess Amazon product reviews
- Visualize rating distributions and unique reviewer counts
- Generate recommendations by collaborative filtering using SVD (Singular Value Decomposition) model

## Usage

To change the input dataset, replace the 'dataset.csv' file with your new dataset (must be in .csv). Then open main.py, change local parameters *datafile*, *columns*, *irrelevance* of the function main(). *datafile* is the name of the new dataset ; *columns* are its columns ; *irrelevance* are columns that are not in the list ['userId', 'productId', 'rating']

To run the recommendation system, execute the following command in your terminal:

python main.py

Make sure that the dataset.csv file is located in the data directory.


## Data

The project relies on an Amazon review dataset, which should be placed in the data folder. The dataset.csv file contains product reviews along with associated metadata necessary for generating recommendations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or features you'd like to add.



"# Recommendation_System" 
