# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class GooglePlayStoreApps:
    def __init__(self, data):
        self.data = data
        self.valid_plot_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']

    def plot_features(self, plot_types):
        """
        Plot features per class label using matplotlib, and it plots the plot_types specified in plot_types.

        Parameters:
        - plot_types (array): An array specifying the type of plot to generate for each feature.

        Supported plot types:
        - 'hist': Histogram
        - 'violin': Violin plot
        - 'box': Box plot
        - 'scatter': Scatter plot
        - 'lines': Line plot
        - 'bar': bar plot
        - 'lollypops': Lollipop plot
        """

        for plot_type in plot_types:
            # Check if the selected plots are in the list of available plots
            if plot_type not in self.valid_plot_types:
                raise ValueError(
                    f"Invalid plot type: {plot_type}. Supported plot types: {', '.join(self.valid_plot_types)}")

        # For each plot, need to produce a subplot for each feature, and then, for the current subplot, check the data
        # related to each label

        for plot_type in plot_types:
            # Create a figure with a single row and subplots for each feature in differnt columns, with width of 15
            # inches and height of 5 inches, fig represents the entire figure, while axes is an array of axes objects
            # representing each subplot (in this case it will be a one-dimensional array containing references to each
            # subplot)
            fig, axes = plt.subplots(nrows=1, ncols=len(self.data.columns), figsize=(15, 5))
            for i, feature in enumerate(self.data.columns):
                # Produce the subplot for each feature
                ax = axes[i]  # One-dimensional array containing references to each subplot

                for label, group in self.data:
                    # Produce a plotting for each label of the current feature
                    ax.grid(True, which='major')

                    # Extract data for the current class label
                    feature_data = group[feature]

                    # Set the title
                    ax.set_title(f'{plot_type.capitalize()} for Feature: {feature}')

                    # Plot according to the specified plot type
                    if plot_type == 'hist':
                        ax.hist(feature_data, bins=10, alpha=0.5, label=label)
                        # Set plot labels
                        ax.set_xlabel(f'{feature}')
                        ax.set_ylabel(f'Frequency')
                        ax.legend()
                    elif plot_type == 'violin':
                        ax.violinplot(feature_data, showmeans=True, showmedians=True, positions=[label])
                        # Set plot title and labels
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Values")
                    elif plot_type == 'box':
                        ax.boxplot(feature_data, vert=True, positions=[label])
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Values")
                    elif plot_type == 'scatter':
                        # Define the x-axis line to go from 0 to len(feature_data) (because first label is 0) then from
                        # then from len(feature_data) to 2*len(feature_data) (because first label is 1) and so one
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.scatter(x_axis, feature_data, label=f'Class {int(label)}')
                        ax.legend()
                    elif plot_type == 'lines':
                        # The standard plot automatically does the x-axis line the same way as done for scatter
                        ax.plot(feature_data, label=f'Class {int(label)}')
                        ax.legend()
                    elif plot_type == 'bar':
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.bar(x_axis, feature_data, label=label)
                        ax.legend()
                    elif plot_type == 'lollypops':
                        x_axis = np.arange(len(feature_data)) + len(feature_data) * label
                        ax.stem(x_axis, feature_data, linefmt='-', label=label)
                        ax.legend()
            # Adjust layout
            plt.tight_layout()
            # Show plot
            plt.show()

    def interpolate(self):
        # Convert 'Category' to integers using LabelEncoder
        label_encoder = LabelEncoder()
        self.data['Category'] = label_encoder.fit_transform(self.data['Category'])

        label_encoder = LabelEncoder()
        self.data['Content Rating'] = label_encoder.fit_transform(self.data['Content Rating'])

        # Option 2: Interpolate missing values (only valid if done withing samples of the same class to be tested,
        # for example, island)
        return self.data.interpolate()

    def remove_outliers(self):
        # Dealing with outliers for all numerical columns
        for column_name in self.data.select_dtypes(include=[np.number]).columns:
            # Calculate the IQR (InterQuartile Range)
            q1 = self.data[column_name].quantile(0.25)
            q3 = self.data[column_name].quantile(0.75)

            # Define the lower and upper bounds to identify outliers
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Remove outliers
            self.data[column_name] = np.where(
                (self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound),
                np.nan,
                self.data[column_name]
            )
        return self.data

    def impute(self):
        # Impute missing values using the mean value of the column
        return self.data.fillna(self.data.iloc[:, :].mean())

    def clean_data(self):
        # Remove lines that are duplicated or that have missing values
        # self.data = self.impute()
        # self.data = self.data.remove_outliers()
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        return self.data

    def assess_correlation(self):
        """
        Assess the correlation between variables.

        This method calculates and displays the correlation matrix (Pearson correlation
        coefficient) between the variables in the dataset, both overall and per class label.
        """

        # Display correlation matrix without considering the class label
        print("\nCorrelation between variables (Pearson correlation coefficient):", self.data.corr())

    def feature_engineering(self):
        # Simple combinations
        self.data['sepal_length_width_product'] = self.data['sepal length (cm)'] * self.data['sepal width (cm)']
        # Complex combination with a nonlinear interaction
        self.data['nonlinear'] = self.data['sepal length (cm)'] ^ 2 + self.data['petal length (cm)'] * self.data[
            'petal width (cm)'] + self.data['petal length (cm)'] * self.data['sepal width (cm)']
        return self.data

    def standardization(self):
        scaler_standard = StandardScaler()
        return scaler_standard.fit_transform(self.data)

    def min_max_scaler(self):
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.data)


if __name__ == "__main__":
    dataset = pd.read_csv('.\data\Google-Playstore.csv')
    gpsa = GooglePlayStoreApps(dataset[
                                   ['Category', 'Rating', 'Rating Count', 'Minimum Installs',
                                    'Maximum Installs', 'Free', 'Price', 'Currency', 'Size', 'Minimum Android',
                                    'Released', 'Last Updated', 'Content Rating', 'Ad Supported', 'In App Purchases']])
    interpolated_data = gpsa.interpolate()
    clean_data = gpsa.clean_data()
    clean_data.assess_correlation()
    # feature_engineering_data = clean_data.feature_engineering()
    # standardized_data = feature_engineering_data.standardization()
    # scaled_data = standardized_data.min_max_scaler()
