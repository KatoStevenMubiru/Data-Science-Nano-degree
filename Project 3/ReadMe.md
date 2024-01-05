

# Udacity Data Science Nanodegree Project: Recommendations with IBM

## Overview
This project is a part of the Udacity Data Science Nanodegree program and focuses on building recommendation systems for the IBM Watson Studio platform. The goal is to provide users with relevant article recommendations based on their interactions with the platform.

## Project Structure
The project is structured as a Jupyter Notebook, and it is divided into several parts, each addressing specific tasks related to recommendation systems.

### Part I: Exploratory Data Analysis
In this section, the notebook explores the dataset, which includes user interactions with articles on the IBM Watson Studio platform. The analysis involves understanding the distribution of user-article interactions and exploring patterns in user behavior.

### Part II: Rank-Based Recommendations
The notebook implements a basic recommendation system based on the popularity of articles. The function `get_top_articles` is created to provide recommendations to users based on the overall popularity of articles.

### Part III: User-User Based Collaborative Filtering
This section involves building a collaborative filtering recommendation system, specifically user-user based. The notebook provides functions for finding similar users and recommending articles based on the preferences of similar users.

### Part IV: Content-Based Recommendations (Extra)
An optional part of the project involves creating a content-based recommendation system. This method leverages the content of articles, such as their body, description, or full name, to provide recommendations. The notebook provides a function `make_content_recs` that allows for experimentation with content-based recommendations.

### Part V: Matrix Factorization
The final part explores matrix factorization using Singular Value Decomposition (SVD) to make article recommendations. The notebook addresses the challenge of choosing the number of latent features and evaluates the model's accuracy on training and test sets.

## Dependencies
The project utilizes the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Running the Notebook
To run the Jupyter Notebook, the user needs to have the above-mentioned libraries installed. The notebook should be executed in a Jupyter environment. 

### Additional Note
This project drew inspiration and guidance from the repository by Stephanie Irvine: [Udacity-Data-Scientist-Nanodegree](https://github.com/stephanieirvine/Udacity-Data-Scientist-Nanodegree/tree/main). Some code snippets and structural ideas were adapted from this source.

