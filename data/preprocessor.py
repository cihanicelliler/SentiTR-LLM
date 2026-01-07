import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from utils.normalization import clean_text
from utils.logger import setup_logger
import glob

logger = setup_logger("Preprocessor")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FSMTSAD")

def load_and_preprocess_data(test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42) -> DatasetDict:
    """
    Loads the FSMTSAD dataset, cleans the text, and splits it into train/val/test.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed.
        
    Returns:
        DatasetDict: Hugging Face DatasetDict containing train, validation, and test splits.
    """
    # Find the dataset file (assuming Excel or CSV)
    # Search recursively for csv or xlsx files
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.xlsx"), recursive=True) + glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {DATA_DIR}. Please run downloader.py first.")
    
    # Heuristic: pick the largest file or specific name if known. 
    data_file = files[0]
    logger.info(f"Loading data from {data_file}...")
    
    if data_file.endswith(".xlsx"):
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file)
        
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Based on inspection, columns are likely 'sentence' and 'label'
    text_col = next((c for c in df.columns if 'sentence' in c or 'text' in c or 'icerik' in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'durum' in c), None)
    
    if not text_col or not label_col:
        logger.warning(f"Could not automatically identify text/label columns. Columns found: {df.columns}")
        # Fallback: assume first column is text, second is label if not found
        text_col = df.columns[0]
        label_col = df.columns[1]
        
    logger.info(f"Using '{text_col}' as text and '{label_col}' as label.")
    
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]]
    
    # Drop NaNs
    df = df.dropna()
    
    # Apply cleaning
    logger.info("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)
    
    # Normalize labels if necessary
    # Paper says: -1 (Negative), 0 (Neutral), 1 (Positive)
    # We want 0, 1, 2 for HF training usually, or keep as is if Regression. 
    # Classification is standard. Let's map to 0: Negative, 1: Neutral, 2: Positive
    # Or strict mapping: -1 -> 0, 0 -> 1, 1 -> 2
    
    unique_labels = sorted(df["label"].unique())
    logger.info(f"Unique labels found: {unique_labels}")
    
    label_map = {}
    if -1 in unique_labels:
        label_map = {-1: 0, 0: 1, 1: 2} # Negative, Neutral, Positive
    elif 'Negative' in unique_labels:
         label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
         
    if label_map:
        df["label"] = df["label"].map(label_map)
        logger.info(f"Mapped labels using: {label_map}")

    # Split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=random_state, stratify=train_df["label"]) # Adjust split base
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    
    return dataset

if __name__ == "__main__":
    try:
        ds = load_and_preprocess_data()
        print(ds)
        # Verify first example
        print("Example:", ds['train'][0])
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
