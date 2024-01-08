#!/usr/bin/env python

import sys
import pandas as pd
from sqlalchemy import create_engine
import argparse

# Function to load data from CSV files
def load_data(messages_filepath, categories_filepath):
    # Read messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge datasets using the common id
    df = pd.merge(messages, categories, on='id')
    return df

# Function to clean data
def clean_data(df):
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df

# Function to save the clean data to an SQLite database
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    # Save the clean dataset into an SQLite database
    df.to_sql('MessageCategories', engine, index=False, if_exists='replace')

# Main function that will run the ETL pipeline
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETL pipeline for disaster response data')
    parser.add_argument('messages_filepath', type=str, help='Filepath for the messages dataset')
    parser.add_argument('categories_filepath', type=str, help='Filepath for the categories dataset')
    parser.add_argument('database_filepath', type=str, help='Filepath for the SQLite database to store the cleaned data')
    
    args = parser.parse_args()

    # Load data
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(args.messages_filepath, args.categories_filepath))
    df = load_data(args.messages_filepath, args.categories_filepath)

    # Clean data
    print('Cleaning data...')
    df = clean_data(df)

    # Save data to database
    print('Saving data...\n    DATABASE: {}'.format(args.database_filepath))
    save_data(df, args.database_filepath)

    # Print success message
    print('Cleaned data saved to database!')

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()