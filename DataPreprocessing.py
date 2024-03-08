from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessing:
    """
    Class responsible for preprocessing the loaded dataset. Need to pass the data with first columns as numerical and
    the last columns as categorical, then indicate how many categorical features are present in the dataset

    Methods: _normalize_features(): Normalizes all features using standard scaling for numerical and min-max scaling
    for categorical.
    """

    def __init__(self, data_loader, number_categorical_features):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.number_categorical_features = number_categorical_features

        # Preprocess data
        self._normalize_features()

    def _normalize_features(self):
        """
        Normalizes features using standard scaling for numerical and min-max scaling for categorical.
        """
        try:
            if self.data_loader.df is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify real numerical features (excluding the last two columns we know are categorical)
            real_numerical_features = self.data_loader.df.columns[:-self.number_categorical_features]

            # Normalize real numerical features using StandardScaler
            scaler = StandardScaler()
            self.data_loader.df[real_numerical_features] = scaler.fit_transform(
                self.data_loader.df[real_numerical_features])

            # Identify encoded features
            encoded_features = self.data_loader.df.columns[-self.number_categorical_features:]

            # Normalize encoded features using MinMaxScaler
            scaler = MinMaxScaler()
            self.data_loader.df[encoded_features] = scaler.fit_transform(
                self.data_loader.df[encoded_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)
