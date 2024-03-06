import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


class FeatureAnalysis:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def perform_pca(self, number_categorical_features, explained_variance_threshold=0.8, plot_pca=True, add_pca=True):
        """
        Perform Principal Component Analysis (PCA) on the numerical features. Can only analyze
        a dataset with all categorical features at the last columns to be ignored

        Parameters:
            n_components (int): Number of principal components to retain.
            number_categorical_features (int): Number of categorical features present in the data
            plot_pca (bool): Defines if the user pretends to plot the produced pca in the train data
            add_pca (bool): Defines if the user pretends to add the produced pca to the original train and test data


        Returns:
            pca_components_train (DataFrame): DataFrame containing the principal components of the train data.
            pca_components_test (DataFrame): DataFrame containing the principal components of the test data.
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None and self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Perform PCA
            pca = PCA()
            pca.fit(self.data_loader.data_train.iloc[:, :-number_categorical_features])

            # Determine the number of components to retain
            explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(explained_variance_ratio_cumulative >= explained_variance_threshold) + 1

            # Perform PCA with the determined number of components
            pca = PCA(n_components=n_components)
            pca_components_train = pca.fit_transform(self.data_loader.data_train.iloc[:, :-number_categorical_features])
            pca_components_train = pd.DataFrame(data=pca_components_train,
                                                columns=[f"PC{i}" for i in range(1, n_components + 1)])
            pca_components_test = pca.transform(self.data_loader.data_test.iloc[:, :-number_categorical_features])
            pca_components_test = pd.DataFrame(data=pca_components_test,
                                               columns=[f"PC{i}" for i in range(1, n_components + 1)])

            if plot_pca:
                self._plot_pca_components(pca_components_train)
            if add_pca:
                self._add_pca_components(pca_components_train, pca_components_test)

            print("PCA performed successfully.")
            return pca_components_train, pca_components_test

        except ValueError as ve:
            print("Error:", ve)

    def _plot_pca_components(self, pca_components):
        """
        Plot all possible combinations of PCA components.

        Parameters:
            pca_components (DataFrame): DataFrame containing the principal components.
        """
        num_components = pca_components.shape[1]
        combinations = list(itertools.combinations(range(num_components), 2))

        fig, axes = plt.subplots(len(combinations), 1, figsize=(8, 5 * len(combinations)))

        for i, (component1, component2) in enumerate(combinations):
            ax = axes[i] if len(combinations) > 1 else axes

            ax.scatter(pca_components.iloc[:, component1], pca_components.iloc[:, component2])
            ax.set_title(f"PCA Component {component1 + 1} and PCA Component {component2 + 1}")
            ax.set_xlabel(f"PCA Component {component1 + 1}")
            ax.set_ylabel(f"PCA Component {component2 + 1}")

        plt.tight_layout()
        plt.show()

    def _add_pca_components(self, pca_components_train, pca_components_test):
        """
        Add PCA components to the original datasets.

        Parameters:
            pca_components_train (DataFrame): DataFrame containing the principal components for the training dataset.
            pca_components_test (DataFrame): DataFrame containing the principal components for the test dataset.
        """

        # Generate column names for PCA components
        pca_column_names = [f"PCA_{i}" for i in range(1, pca_components_train.shape[1] + 1)]

        # Assign column names to PCA components
        pca_components_train.columns = pca_column_names
        pca_components_test.columns = pca_column_names

        # Set indices to match before concatenating
        pca_components_train.index = self.data_loader.data_train.index
        pca_components_test.index = self.data_loader.data_test.index

        # Concatenate the PCA components with the original datasets
        self.data_loader.data_train = pd.concat([self.data_loader.data_train, pca_components_train], axis=1)
        self.data_loader.data_test = pd.concat([self.data_loader.data_test, pca_components_test], axis=1)

    def relevant_feature_identification(self, num_features=10):
        """
        Perform feature relevant feature identification using mutual information between each feature and the target
        variable. Mutual information measures the amount of information obtained about one random variable through
        another random variable. It quantifies the amount of uncertainty reduced for one variable given the knowledge
        of another variable. In feature selection, mutual information helps identify the relevance of features with
        respect to the target variable.

        Parameters:
            num_features (int): Number of features to select.

        Returns:
            selected_features (list): List of selected feature names.
        """
        try:
            # Check if data_train is not None
            if self.data_loader.data_train is None or self.data_loader.labels_train is None:
                raise ValueError("Training data or labels have not been loaded yet.")

            # Perform feature selection using mutual information
            mutual_info = mutual_info_classif(self.data_loader.data_train, self.data_loader.labels_train)

            selected_features_indices = np.argsort(mutual_info)[::-1][:num_features]
            selected_features = self.data_loader.data_train.columns[selected_features_indices]

            print(f"{num_features} relevant features identified.")
            print(selected_features.tolist())
            return selected_features.tolist()

        except ValueError as ve:
            print("Error:", ve)
