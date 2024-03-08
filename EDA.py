import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


class EDA:
    """
    A class responsible for exploratory data analysis (EDA).

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_eda(): Performs exploratory data analysis.
        plot_distributions(): Plots distributions of the features.
        plot_correlation_heatmap(): Plots a correlation heatmap between features and labels.
        plot_feature_importance(): Computes and visualizes feature importance using permutation importance.
    """

    def __init__(self, data_loader):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.data_loader = data_loader

    def perform_eda(self):
        """
        Performs exploratory data analysis.
        """
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Summary statistics
        print("\nSummary Statistics for dataframe:")
        print(self.data_loader.df.describe())

        df_copy = self.data_loader.df.iloc[:-1659000, :-16].copy()

        # Distribution analysis
        self.plot_distributions(df_copy)

        # Correlation analysis
        self.plot_correlation_heatmap(df_copy)

        # Feature Importance analysis
        X_train, X_test, y_train = self.plot_feature_importance(df_copy)

        return X_train, X_test, y_train

    def plot_distributions(self, df_copy):
        """
        Plots distributions of the features.
        """

        num_cols = len(df_copy.columns)
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))
        for i, feature in enumerate(df_copy.columns):
            sns.histplot(data=df_copy, x=feature, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df_copy):
        """
        Plots a correlation heatmap between features and labels.
        """

        # Compute the correlation matrix
        corr_matrix = df_copy.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap between Features and Labels")
        plt.show()

    def plot_feature_importance(self, df_copy, n_estimators=5, n_repeats=2):
        """
        Computes and visualizes feature importance using permutation importance.
        """
        X = self.data_loader.df[['Category', 'Rating', 'Rating Count', 'RatingType']].drop(['RatingType'], axis=1)
        y = self.data_loader.df['RatingType']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

        # Fit a random forest classifier to compute feature importance
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        clf_acc = accuracy_score(y_pred, y_test) * 100
        print("Accuracy =", round(clf_acc, 2), "%")
        cm = confusion_matrix(y_pred, y_test)

        # Get unique classes from y_pred and y_test
        unique_classes = np.unique(np.concatenate((y_pred, y_test)))
        cmd = ConfusionMatrixDisplay(cm, display_labels=unique_classes)

        fig, ax = plt.subplots(figsize=(12, 12))
        plt.title("Confusion Matrix RandomForestClassifier")
        cmd.plot(ax=ax)
        plt.show()

        # Compute permutation importance
        result = permutation_importance(clf, X_train, y_train, n_repeats=n_repeats)
        sorted_idx = result.importances_mean.argsort()

        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.show()

        return X_train, X_test, y_train
