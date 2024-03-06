# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class GooglePlayStoreApps:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Remove lines that are duplicated or that have missing values
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        return self.data

    def data_interpolation(self):
        # Convert 'Category' to integers using LabelEncoder
        label_encoder = LabelEncoder()
        self.data['Category'] = label_encoder.fit_transform(self.data['Category'])

        label_encoder = LabelEncoder()
        self.data['Content Rating'] = label_encoder.fit_transform(self.data['Content Rating'])

        # Option 2: Interpolate missing values (only valid if done withing samples of the same class to be tested,
        # for example, island)
        return self.data.interpolate()

    def data_standardization(self):
        scaler_standard = StandardScaler()
        return scaler_standard.fit_transform(self.data)

    def data_minmaxscaler(self):
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.data)


if __name__ == "__main__":
    dataset = pd.read_csv('.\data\Google-Playstore.csv')
    gpsa = GooglePlayStoreApps(dataset[
                                   ['App Name', 'Category', 'Rating', 'Rating Count', 'Minimum Installs',
                                    'Maximum Installs', 'Free', 'Price', 'Currency', 'Size', 'Minimum Android',
                                    'Released', 'Last Updated', 'Content Rating', 'Ad Supported', 'In App Purchases']])
    interpolated_data = gpsa.data_interpolation()
    clean_data = gpsa.clean_data()
    standardized_data = gpsa.data_standardization()
    scaled_data = gpsa.data_minmaxscaler()
