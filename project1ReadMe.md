# Airbnb Boston Data Analysis
# This the link to the blog post 
https://medium.com/@kato.steven60/unraveling-the-dynamics-of-airbnb-in-boston-a-data-driven-exploration-cc6de7f1c673


## Overview
This project explores the Airbnb activity in Boston using a dataset obtained from Airbnb Inside initiative. The dataset includes information about listings, reviews, and calendar availability. The goal is to derive insights into the busiest times to visit Boston, pricing patterns, and predicting the popularity of listings.

## Motivation
The motivation behind this project is to provide valuable information for both potential guests and hosts in Boston. By analyzing the data, we aim to uncover trends and patterns that can assist in making informed decisions about travel plans, pricing strategies for hosts, and predicting the popularity of listings.

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Files in the Repository

### 1. Jupyter Notebooks
- `1_data_exploration.ipynb`: Explore the data, perform descriptive statistics, and visualize key insights.
- `2_pricing_analysis.ipynb`: Analyze pricing patterns based on different factors.
- `3_prediction_model.ipynb`: Build and evaluate a predictive model for listing popularity.

### 2. Data Files
- `calendar.csv`: Calendar data including listing availability and pricing.
- `listings.csv`: Detailed information about the listings, including features and pricing.
- `reviews.csv`: Dataset containing reviews, reviewer information, and comments.

### 3. README.md
You are reading it right now! It provides an overview of the project, motivation, used libraries, and information about the files in the repository.

## Summary of Results

### Busiest Times to Visit Boston
- Identified peak periods of demand and corresponding pricing spikes.

### Pricing Patterns
- Explored pricing variations based on factors such as property type, room type, and neighborhood.

### Predicting Listing Popularity
- Utilized a linear regression model to predict listing popularity.
- Evaluation Metrics:
  - Mean Squared Error: 1221.95
  - R-squared Score: 0.0695
  - Cross-Validation Scores: [-0.0338, -0.1735, -0.0216, -0.0362, -0.1189]

## Acknowledgements
- This project was completed as part of [mention your course/program].
- The dataset was obtained from Airbnb Inside initiative.
- References to online resources, including StackOverflow and Kaggle, were consulted during the project.

Feel free to explore the Jupyter notebooks for a detailed walkthrough of the analysis and findings!

