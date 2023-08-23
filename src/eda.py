import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import random

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
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
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

        plt.show()

    # Function to display class distribution
    def display_class_distribution(self, dataset):
        # Count the number of samples in each class
        class_counts = torch.bincount(dataset.targets, minlength=10)
        
        # Create a bar plot to visualize the class distribution
        plt.figure(figsize=(10, 6))
        plt.bar(self.class_labels, class_counts, color='#ff6900')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution', fontweight='bold')
        plt.gca().spines['top'].set_visible(False)    
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        plt.show()
    
    # Function to display average black pixels
    def display_average_black_pixels(self, dataset):
        # Calculate the average number of black pixels per class
        class_pixel_averages = []

        for class_label in range(10):
            class_indices = torch.where(dataset.targets == class_label)[0]
            class_images = dataset.data[class_indices]
            class_black_pixels = class_images.eq(0).sum(dim=(1, 2))
            class_pixel_averages.append(class_black_pixels.float().mean().item())

        # Create a bar plot to visualize the average black pixels per class
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_labels, class_pixel_averages, color='#ff6900')
        plt.xlabel('Class')
        plt.ylabel('Average Black Pixels')
        plt.yticks([]) 
        plt.title('Average Black Pixels per Class in Fashion MNIST', fontweight='bold')
        plt.xticks(rotation=45)
        plt.gca().spines['top'].set_visible(False)    
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Add pixel count labels on top of the bars
        for bar, pixel_count in zip(bars, class_pixel_averages):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pixel_count:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

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