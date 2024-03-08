import pandas as pd
from sklearn.preprocessing import LabelEncoder

from DataLoader import DataLoader


class DataManipulator(DataLoader):
    def __init__(self, filename, drop_columns):
        """
        Initializes the DataManipulator with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        super().__init__(filename, drop_columns)

        # Manipulate data
        self._price_range()
        self._revenue()
        self._estimated_revenue()
        self._rating_type()
        self._max_installs_per_rating()
        self._in_app_purchases_per_content_rating()
        self._price_per_content_rating()
        self._max_installs_per_content_rating()
        self._in_app_purchases_per_category()
        self._ad_supported_per_category()

    def _price_range(self):
        self.df['PriceRange'] = pd.cut(self.df['Price'], bins=[0, 0.19, 9.99, 29.99, 410],
                                       labels=['Free', 'Low', 'Mid', 'High'],
                                       include_lowest=True)

    def _revenue(self):
        self.df['Revenue'] = self.df['Price'] * self.df['Maximum Installs']

    def _estimated_revenue(self):
        encoder = LabelEncoder()
        self.df['In App Purchases'] = encoder.fit_transform(self.df['In App Purchases'])
        self.df['Ad Supported'] = encoder.fit_transform(self.df['Ad Supported'])
        self.df['EstimatedRevenue'] = (self.df['In App Purchases'] + self.df['Ad Supported']) * self.df[
            'Maximum Installs'] / 2 + self.df['Revenue']

    def _rating_type(self):
        self.df['RatingType'] = 'NoRating'
        self.df.loc[(self.df['Rating Count'] > 0) & (self.df['Rating Count'] <= 10000.0), 'RatingType'] = 'Less than 10K'
        self.df.loc[(self.df['Rating Count'] > 10000) & (
                self.df['Rating Count'] <= 500000.0), 'RatingType'] = 'Between 10K and 500K'
        self.df.loc[(self.df['Rating Count'] > 500000) & (
                self.df['Rating Count'] <= 138557570.0), 'RatingType'] = 'More than 500K'

    def _max_installs_per_rating(self):
        self.df['MaxInstalls_Rating'] = self.df['Rating'] * self.df['Maximum Installs']

    def _in_app_purchases_per_content_rating(self):
        encoder = LabelEncoder()
        self.df['Content Rating'] = self.df['Content Rating'].replace('Unrated', "Everyone")
        self.df['Content Rating'] = self.df['Content Rating'].replace('Mature 17+', "Adults")
        self.df['Content Rating'] = self.df['Content Rating'].replace('Adults only 18+', "Adults")
        self.df['Content Rating'] = self.df['Content Rating'].replace('Everyone 10+', "Everyone")
        self.df['Content Rating'] = encoder.fit_transform(self.df['Content Rating'])
        self.df['InAppPurchases_ContentRating'] = self.df['In App Purchases'] * self.df['Content Rating']

    def _price_per_content_rating(self):
        self.df['Price_ContentRating'] = self.df['Content Rating'] * self.df['Price']

    def _max_installs_per_content_rating(self):
        self.df['MaxInstalls_ContentRating'] = self.df['Content Rating'] * self.df['Maximum Installs']

    def _in_app_purchases_per_category(self):
        encoder = LabelEncoder()
        self.df['Category'] = encoder.fit_transform(self.df['Category'])
        self.df['InAppPurchases_Category'] = self.df['In App Purchases'] * self.df['Category']

    def _ad_supported_per_category(self):
        self.df['AdSupported_Category'] = self.df['Ad Supported'] * self.df['Category']
