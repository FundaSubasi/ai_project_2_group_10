# Importing dependencies
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data explanation placeholder
# This script processes movie and economics data, combines them, and prepares the data for machine learning.

def read_data(movies_path: str, economics_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads movie and economics data from CSV files.

    Args:
        movies_path (str): Path to the movies data CSV file.
        economics_path (str): Path to the economics data CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for movies and economics data.
    """
    try:
        df_movies = pd.read_csv(movies_path)
        df_economics = pd.read_csv(economics_path)
        logging.info("Data successfully read.")
        return df_movies, df_economics
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        raise

def preprocess_data(df_movies: pd.DataFrame, df_economics: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and combines movie and economics data.

    Args:
        df_movies (pd.DataFrame): DataFrame containing movie data.
        df_economics (pd.DataFrame): DataFrame containing economics data.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    try:
        # Create a 'Date' column for datetime index from year, month, and day
        df_movies['Date'] = pd.to_datetime({
            'year': df_movies['released_year'],
            'month': df_movies['released_month'],
            'day': df_movies['released_day']
        })
        # Set 'Date' as the index and sort the DataFrame by date
        df_movies.set_index('Date', inplace=True)
        df_movies.sort_index(inplace=True)

        # Extract 'Year' and 'Month' from 'Date' column in the economics data
        df_economics['Year'] = df_economics['Date'].str.slice(0, 4).astype(int)
        df_economics['Month'] = df_economics['Date'].str.slice(5, 7).astype(int)

        # Rename movie columns for consistency
        df_movies.rename(columns={
            'released_year': 'Year',
            'released_month': 'Month'
        }, inplace=True)

        logging.info(f'Total economic records: {df_economics.shape[0]}')
        logging.info(f'Total movie records: {df_movies.shape[0]}')

        # Merge the movies and economics data on 'Year' and 'Month'
        df_combined = pd.merge(df_economics, df_movies, how='left', on=['Year', 'Month'])
        logging.info(f'Total records after concatenation: {df_combined.shape[0]}')

        return df_combined
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

def create_target_and_drop_features(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a target column and drops unnecessary features.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame with target column and dropped features.
    """
    # Create 'Target' column by concatenating 'critical_success', 'financial_success', and 'Economic Climate'
    df_combined['Target'] = df_combined['critical_success'] + ' ' + \
                            df_combined['financial_success'] + ' ' + \
                            df_combined['Economic Climate']
    # Display counts of unique values in the 'Target' column
    df_combined['Target'].value_counts()

    # Columns to drop as they are not needed for modeling
    cols_to_drop = [
        'Economic Climate',
        'Year',
        'Month',
        'id',
        'critical_success',
        'financial_success',
        'released_day'
    ]
    # Drop the unnecessary columns
    df_combined.drop(columns=cols_to_drop, inplace=True)
    return df_combined

def split_and_scale_data(df_combined: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and testing sets, and scales numerical features.

    Args:
        df_combined (pd.DataFrame): Combined and processed DataFrame.
        target_col (str): Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Scaled and encoded training and testing data.
    """
    # Columns that need to be scaled
    col_to_scale = [
        'CCI Value', 'CCI Rolling Mean', 'CCI Rolling Percent Change',
        'CPI Value', 'CPI Rolling Mean', 'CPI Rolling Percent Change',
        'Unemployment Rate (%)', 'Unemployment Rate (%) Rolling Mean',
        'Unemployment Rate Rolling Percent Change', 'vote_average', 'vote_count',
        'revenue', 'runtime', 'budget', 'popularity', 'roi'
    ]

    # Columns that need to be encoded
    col_to_encode = [
        'Date', 'CCI Rolling Percent Change Flag', 'CPI Rolling Percent Change Flag',
        'Unemployment Rate Rolling Percent Change Flag', 'title',
        'status', 'release_date', 'original_language', 'original_title',
        'genres', 'production_companies', 'production_countries',
        'spoken_languages', 'cast', 'director', 'writers', 'producers'
    ]

    # Separate features (X) and target (y)
    X = df_combined.drop(columns=target_col)
    y = df_combined[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[col_to_scale])
    X_test_scaled = scaler.transform(X_test[col_to_scale])

    # Convert scaled features back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=scaler.get_feature_names_out())
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=scaler.get_feature_names_out())

    # Encode categorical features
    encoder_x = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder_x.fit(X_train[col_to_encode])

    encoder_y = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder_y.fit(y_train.values.reshape(-1, 1))

    X_train_encoded = encoder_x.transform(X_train[col_to_encode])
    X_test_encoded = encoder_x.transform(X_test[col_to_encode])

    y_train_encoded = encoder_y.transform(y_train.values.reshape(-1, 1))
    y_test_encoded = encoder_y.transform(y_test.values.reshape(-1, 1))

    # Convert encoded features back to DataFrame
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_x.get_feature_names_out())
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_x.get_feature_names_out())

    # Concatenate scaled and encoded features
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis=1)
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis=1)

    return X_train_final, X_test_final, y_train_encoded, y_test_encoded

def main():
    # Paths to data
    movies_path = "./Resources/movies_data.csv"
    economics_path = "./Resources/economics_data.csv"

    # Reading in data
    df_movies, df_economics = read_data(movies_path, economics_path)

    # Preprocessing and combining data
    df_combined = preprocess_data(df_movies, df_economics)

    # Creating target and dropping unnecessary features
    df_combined = create_target_and_drop_features(df_combined)

    # Splitting and scaling data
    X_train, X_test, y_train, y_test = split_and_scale_data(df_combined, 'Target')

    logging.info("Data processing complete.")

if __name__ == "__main__":
    main()
