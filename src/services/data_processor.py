import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Split the indexed dataset into train/validation/test sets."""

    def __init__(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
        self.df = df
        self.X = X
        self.y = y


    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """Split data into train, validation, and test sets."""
        if self.X is None or self.y is None:
            if self.df is None or self.df.empty:
                raise ValueError("Data not loaded. Call load_data() or set_data() first.")
            # Fallback to df
            X = self.df.drop(columns=['label'])
            y = self.df['label']
        else:
            X, y = self.X, self.y

        # First split: train + val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_labels(self) -> list:
        """Get unique labels from the dataset."""
        if self.df is not None and not self.df.empty:
            return self.df['label'].unique().tolist()
        elif self.y is not None:
            return list(set(self.y))
        return []






    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """Load the indexed dataset from a CSV or DataFrame."""
        if csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            # Assume it's passed or built elsewhere; for now, return empty
            self.df = pd.DataFrame()
        return self.df

    def set_data(self, X, y):
        """Set preprocessed data directly."""
        self.X = X
        self.y = y
