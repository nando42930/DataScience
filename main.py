from DataLoader import DataLoader
from DataVisualization import DataVisualization


class GooglePlayStoreApps:
    def __init__(self):
        self.data = DataLoader(filename='.\data\Google-Playstore.csv')
        self.plot_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']

    def __main__(self):
        DataVisualization(self.data, plot_types=self.plot_types)

    # def assess_correlation(self):
    #     """
    #     Assess the correlation between variables.
    #
    #     This method calculates and displays the correlation matrix (Pearson correlation
    #     coefficient) between the variables in the dataset, both overall and per class label.
    #     """
    #
    #     # Display correlation matrix without considering the class label
    #     print("\nCorrelation between variables (Pearson correlation coefficient):", self.data.corr())
    #
    # def feature_engineering(self):
    #     # Simple combinations
    #     self.data['sepal_length_width_product'] = self.data['sepal length (cm)'] * self.data['sepal width (cm)']
    #     # Complex combination with a nonlinear interaction
    #     self.data['nonlinear'] = self.data['sepal length (cm)'] ^ 2 + self.data['petal length (cm)'] * self.data[
    #         'petal width (cm)'] + self.data['petal length (cm)'] * self.data['sepal width (cm)']
    #     return self.data


# gpsa = GooglePlayStoreApps(data_loader[['Category', 'Rating', 'Rating Count', 'Minimum Installs', 'Maximum Installs', 'Free', 'Price', 'Currency', 'Size', 'Minimum Android', 'Released', 'Last Updated', 'Content Rating', 'Ad Supported', 'In App Purchases']])