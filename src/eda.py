import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

class EDA:
    def __init__(self) -> None:
        self.load_data()
        self.class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Function load fashion MNIST data
    def load_data(self, data_dir="./data"):
        # transforms
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # datasets
        self.trainset = torchvision.datasets.FashionMNIST(data_dir,
            download=True,
            train=True,
            transform=transform)
        self.testset = torchvision.datasets.FashionMNIST(data_dir,
            download=True,
            train=False,
            transform=transform)
        
        return self.trainset, self.testset
    
    # Function to display random images with labels
    def display_random_images(self, num_images=5):
        # Display a random image from each class
        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(9,4))
        fig.tight_layout()

        for class_label in range(10):
            class_indices = torch.where(self.trainset.targets == class_label)[0]
            random_index = random.choice(class_indices)
            random_image = self.trainset[random_index][0].numpy()
            ax = axes[class_label // cols, class_label % cols]
            ax.imshow(random_image[0], cmap='gray')
            ax.set_title(self.class_labels[class_label])
            ax.set_title(self.class_labels[class_label], fontdict={'color': '#ff6900'})
            ax.axis('off')
        
        # plt.savefig('random_images', bbox_inches='tight')
        plt.show()

    # Function to display class distribution
    def plot_class_distribution(self, trainset, testset):
        # Compute class counts for the training and test sets
        train_class_counts = torch.bincount(trainset.targets, minlength=10)
        test_class_counts = torch.bincount(testset.targets, minlength=10)

        # Create bar chart traces for the training and test sets
        trace1 = go.Bar(
            x=trainset.classes,
            y=train_class_counts,
            marker_color='#ff6900',
            name='Train Set'
        )

        trace2 = go.Bar(
            x=testset.classes,
            y=test_class_counts,
            marker_color='#ff6900',
            name='Test Set'
        )

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Train Set', 'Test Set'))

        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)

        fig.update_layout(
            title='Class Distributions for Train and Test Sets',
            xaxis=dict(tickangle=45),
            showlegend=False,
            barmode='group',
            margin=dict(t=80, b=80, l=30, r=20)
        )

        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_layout(height=600)
        fig.update_layout(titlefont=dict(size=16, color='#000000'))
        fig.update_yaxes(tickfont_color='#000000', row=1, col=1)
        fig.update_yaxes(tickfont_color='#000000', row=2, col=1)
        fig.update_xaxes(tickfont_color='#000000', row=1, col=1)
        fig.update_xaxes(tickfont_color='#000000', row=2, col=1)

        # Show the subplot figure
        fig.show()

    # Function to display average black pixels
    def display_average_black_pixels(self, trainset, testset):
        # Calculate the average number of black pixels per class
        class_pixel_averages_train = []
        class_pixel_averages_test = []

        for class_label in range(10):
            class_indices = torch.where(trainset.targets == class_label)[0]
            class_images = trainset.data[class_indices]
            class_black_pixels = class_images.eq(0).sum(dim=(1, 2))
            class_pixel_averages_train.append(class_black_pixels.float().mean().item())

        for class_label in range(10):
            class_indices = torch.where(testset.targets == class_label)[0]
            class_images = testset.data[class_indices]
            class_black_pixels = class_images.eq(0).sum(dim=(1, 2))
            class_pixel_averages_test.append(class_black_pixels.float().mean().item())

        # Create a bar plot to visualize the average black pixels per class
        trace1 = go.Figure(data=[go.Bar(
            x=self.class_labels,
            y=class_pixel_averages_train,
            marker_color='#ff6900',
            text=[f'{pixel_count:.2f}' for pixel_count in class_pixel_averages_train],  # Only the numerical values
            textposition='auto'
        )])

        trace2 = go.Figure(data=[go.Bar(
            x=self.class_labels,
            y=class_pixel_averages_test,
            marker_color='#ff6900',
            text=[f'{pixel_count:.2f}' for pixel_count in class_pixel_averages_test],  # Only the numerical values
            textposition='auto'
        )])

        trace1.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        trace2.update_traces(texttemplate='%{text:.2s}', textposition='inside')

        # make a subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Train Set', 'Test Set'))

        fig.add_trace(trace1.data[0], row=1, col=1)
        fig.add_trace(trace2.data[0], row=2, col=1)

        # update layout
        fig.update_layout(
            title='Average Number of Black Pixels per Class',
            xaxis=dict(tickangle=45),
            showlegend=False,
            margin=dict(t=80, b=80, l=40, r=20)
        )

        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_layout(height=600)
        fig.update_layout(titlefont=dict(size=16, color='#000000'))
        fig.update_yaxes(tickfont_color='#000000', row=1, col=1)
        fig.update_yaxes(tickfont_color='#000000', row=2, col=1)
        fig.update_xaxes(tickfont_color='#000000', row=1, col=1)
        fig.update_xaxes(tickfont_color='#000000', row=2, col=1)

        # change color of bar labels
        for trace in fig.data:
            trace.marker.color = '#ff6900'
            trace.textfont.color = '#000000'
        

        fig.show()

    def _output_label(self, label):
        output_mapping = {
                    0: "T-shirt/Top",
                    1: "Trouser",
                    2: "Pullover",
                    3: "Dress",
                    4: "Coat", 
                    5: "Sandal", 
                    6: "Shirt",
                    7: "Sneaker",
                    8: "Bag",
                    9: "Ankle Boot"
                    }
        input = (label.item() if type(label) == torch.Tensor else label)
        return output_mapping[input]