import numpy as np
from matplotlib import pyplot as plt


class DataVisualization:
    def __init__(self, data_loader, plot_types):
        """
        Initializes the DataVisualization class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.plot_types = plot_types
        self.valid_plot_types = ['hist', 'violin', 'box', 'scatter', 'lines', 'bar', 'lollypops']
        self.plot_features(self.plot_types)

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
            fig, axes = plt.subplots(nrows=1, ncols=len(self.data_loader.columns), figsize=(15, 5))
            for i, feature in enumerate(self.data_loader.columns):
                # Produce the subplot for each feature
                ax = axes[i]  # One-dimensional array containing references to each subplot

                for label, group in self.data_loader:
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
