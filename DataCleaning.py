class DataCleaning:
    """
    Class for cleaning operations.

    Methods:
        remove_duplicates(): Remove duplicate rows from the dataset.
        handle_missing_values(strategy='mean'): Handle missing values using the specified strategy.
        remove_outliers(threshold=3): Remove outliers from the dataset
    """

    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader

    def remove_duplicates(self):
        """
        Remove duplicate rows from the train dataset.
        """
        try:
            # Check if data and labels are not None
            if self.data_loader.df is None:
                raise ValueError("Data has not been loaded yet.")

            num_samples_before = len(self.data_loader.df)
            print("Number of samples before removing duplicates:", num_samples_before)

            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.df.drop_duplicates(inplace=True)

            num_samples_after = len(self.data_loader.df)
            print("Number of samples after removing duplicates:", num_samples_after)
            print()

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values using the specified strategy.

        Parameters:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'most_frequent', or a constant value).
        """
        try:
            # Check if data is not None
            if self.data_loader.df is None:
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.df.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return
            else:
                num_samples_before = len(self.data_loader.df)
                print("Number of samples before handling missing values:", num_samples_before)

            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.df.fillna(self.data_loader.df.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.df.fillna(self.data_loader.df.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.df.fillna(self.data_loader.df.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.df.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.df = self.data_loader.df.dropna(axis=0)
            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)
            num_samples_after = len(self.data_loader.df)
            print("Number of samples after handling missing values:", num_samples_after)
            print()

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self, threshold=4):
        """
        Detect outliers in numerical features using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.

        Returns:
            outliers (DataFrame): DataFrame containing the outliers.
        """
        try:
            # Check if test data is not None
            if self.data_loader.df is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify numerical features
            numerical_features = self.data_loader.df.select_dtypes(include=['number'])

            # Calculate z-scores for numerical features
            z_scores = (numerical_features - numerical_features.mean()) / numerical_features.std()

            # Find outliers based on threshold
            outliers = self.data_loader.df[(z_scores.abs() > threshold).any(axis=1)]

            return outliers

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self, threshold=2):
        """
        Remove outliers from the dataset using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.df is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers(threshold)

            num_samples_before = len(self.data_loader.df)
            print("Number of samples before removing outliers:", num_samples_before)

            # Remove outliers from the dataset
            self.data_loader.df = self.data_loader.df.drop(outliers.index)

            num_samples_after = len(self.data_loader.df)
            print("Number of samples after removing outliers:", num_samples_after)
            print()

        except ValueError as ve:
            print("Error:", ve)
