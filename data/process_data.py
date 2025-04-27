import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them on 'id'.
    
    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.
    
    Returns:
    df: dataframe. Merged dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting values to binary, 
    and removing duplicates.
    
    Args:
    df: dataframe. Merged dataset.
    
    Returns:
    df: dataframe. Cleaned dataset.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Drop rows where any category column has a value of 2. It was suggested by the reviewers, and it makes more sense.
    categories = categories[(categories != 2).all(axis=1)]
    df = df.loc[categories.index]  # Keep only rows with valid categories
    
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save cleaned data into a SQLite database.

    Args:
    df: dataframe. Cleaned data.
    database_filename: str. Filepath for the output SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        # Plot all category value counts in one canvas
        category_columns = df.columns[4:]  # Adjust if your first 4 columns are id, message, original, genre
        n_cols = 4
        n_rows = math.ceil(len(category_columns) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten()

        for i, col in enumerate(category_columns):
            counts = df[col].value_counts().sort_index()
            axes[i].bar(counts.index.astype(str), counts.values)
            axes[i].set_title(col)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks([0, 1])
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('all_category_value_counts.png')
        plt.close()
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()