import pandas as pd


class DataLoader:
    """
        Generic Class responsible for loading the dataset and splitting it into training and testing sets.

        Attributes:
            filename (str): The filename of the dataset to load.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int or None): The seed used by the random number generator for reproducibility.

        Attributes (after loading the data):
            data_train (DataFrame): The training data features.
            labels_train (Series): The training data labels.
            data_test (DataFrame): The testing data features.
            labels_test (Series): The testing data labels.

        Methods:
            _load_data(): Loads the dataset, splits it into training and testing sets,
                          and assigns the data and labels to the appropriate attributes.
        """

    def __init__(self, filename, drop_columns):
        """
        Initializes the DataLoader with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        self.filename = filename
        self.df = None

        # Load data
        self._load_data(drop_columns=drop_columns)

    def _load_data(self, drop_columns):
        """
        Loads the dataset from the specified filename,
        splits it into training and testing sets using train_test_split(),
        and assigns the data and labels to the appropriate attributes.
        """
        try:
            # Load the dataset
            self.df = pd.read_csv(self.filename)
            # Split the data into features and labels
            self.df = self.df.drop(columns=drop_columns)
            print("Data loaded successfully.")
            print()

        except FileNotFoundError:
            print("File not found. Please check the file path.")
