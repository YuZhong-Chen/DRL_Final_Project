import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Node:

    def __init__(self, image: Image, desp: str, point: np.array):
        self.img = image
        self.desp = desp
        self.point = point

    def __str__(self):
        return f"Node: {self.desp} at {np.array2string(self.point, separator=', ')}"
    
    # Get the image
    def get_image(self):
        return self.img
    
    # Get the description
    def get_description(self):
        return self.desp
    
    # Use plt to display the image and description
    def display(self):
        plt.imshow(self.img)
        plt.title(self.desp)
        plt.show()

# Test the node with an image
if __name__ == "__main__":

    # Log the start point
    print("============= Testing the node =============")

    # Load the image
    img = Image.open("./data/img/bed.png")

    # Create a node
    node = Node(img, "Bed", np.array([0, 0]))

    # Print the node
    print(node)

    # Display the node
    node.display()