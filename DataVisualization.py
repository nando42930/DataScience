import numpy as np
from matplotlib import pyplot as plt


class DataVisualization:
    def __init__(self, data_loader):
        """
        Initializes the DataVisualization class with a DataLoader object.
        """
        self.data_loader = data_loader
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
            # Create a figure with a single row and subplots for each feature in different columns, with width of 15
            # inches and height of 5 inches, fig represents the entire figure, while axes is an array of axes objects
            # representing each subplot (in this case it will be a one-dimensional array containing references to each
            # subplot)
            fig, axes = plt.subplots(nrows=1, ncols=len(self.data_loader.df.columns), figsize=(15, 5))
            for i, (feature, feature_data) in enumerate(self.data_loader.df.items()):
                # Produce the subplot for each feature
                ax = axes[i]  # One-dimensional array containing references to each subplot

                ax.grid(True, which='major')

                # Set the title
                ax.set_title(f'{plot_type.capitalize()} for Feature: {feature}')

                # Plot according to the specified plot type
                if plot_type == 'hist':
                    ax.hist(feature_data, bins=10, alpha=0.5)
                    # Set plot labels
                    ax.set_xlabel(f'{feature}')
                    ax.set_ylabel(f'Frequency')
                elif plot_type == 'violin':
                    ax.violinplot(feature_data, showmeans=True, showmedians=True)
                    ax.set_xlabel("Class")
                    ax.set_ylabel("Values")
                elif plot_type == 'box':
                    ax.boxplot(feature_data, vert=True)
                    ax.set_xlabel("Class")
                    ax.set_ylabel("Values")
                elif plot_type == 'scatter':
                    x_axis = np.arange(len(feature_data))
                    ax.scatter(x_axis, feature_data)
                elif plot_type == 'lines':
                    x_axis = np.arange(len(feature_data))
                    ax.plot(x_axis, feature_data)
                elif plot_type == 'bar':
                    x_axis = np.arange(len(feature_data))
                    ax.bar(x_axis, feature_data)
                elif plot_type == 'lollypops':
                    x_axis = np.arange(len(feature_data))
                    ax.stem(x_axis, feature_data, linefmt='-')

            # Adjust layout
            plt.tight_layout()
            # Show plot
            plt.show()
