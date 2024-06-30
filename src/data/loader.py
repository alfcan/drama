import pandas as pd
import os

def load_dataset(file_path):
    """
    Loads a dataset from a CSV file.

    Args:
    - file_path (str): The path to the CSV file to load.

    Returns:
    - pd.DataFrame: The dataset loaded into a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        dataset = pd.read_csv(file_path)
        print(f"Dataset successfully loaded from {file_path}")
        return dataset
    except Exception as e:
        raise ValueError(f"Error while loading the CSV file: {e}")