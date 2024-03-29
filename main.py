from DataCleaning import DataCleaning
from DataManipulator import DataManipulator
from DataPreprocessing import DataPreprocessing
from DataVisualization import DataVisualization
from EDA import EDA
from FeatureAnalysis import FeatureAnalysis
from HypothesisTesting import HypothesisTesting


class GooglePlayStoreApps:
    def __init__(self):
        self.data = DataManipulator(filename='./data/Google-Playstore.csv',
                                    drop_columns=['App Name', 'App Id', 'Installs', 'Minimum Installs', 'Free',
                                                  'Currency', 'Developer Id', 'Developer Website', 'Developer Email',
                                                  'Released', 'Last Updated', 'Privacy Policy', 'Editors Choice',
                                                  'Scraped Time'])

        # Reorder columns
        cols_to_move = ['Size', 'Minimum Android', 'PriceRange', 'RatingType']
        self.data.df = self.data.df[[col for col in self.data.df.columns if col not in cols_to_move] + cols_to_move]

        self.plot_types = ['hist']
        # self.plot_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']

    def run(self):
        DataPreprocessing(self.data, number_categorical_features=0)
        DataCleaning(self.data).remove_duplicates()
        DataCleaning(self.data).handle_missing_values()
        DataCleaning(self.data).remove_outliers()
        DataVisualization(self.data).plot_features(self.plot_types)
        self.data.data_train, self.data.data_test, self.data.labels_train = EDA(self.data).perform_eda()
        FeatureAnalysis(self.data).perform_pca(number_categorical_features=4)
        FeatureAnalysis(self.data).relevant_feature_identification(num_features=3)
        hypothesis_tester = HypothesisTesting(self.data)
        hypothesis_tester.anova_results()
        hypothesis_tester.kruskal_wallis_results()
        hypothesis_tester.t_test_results()


if __name__ == "__main__":
    app = GooglePlayStoreApps()
    app.run()
