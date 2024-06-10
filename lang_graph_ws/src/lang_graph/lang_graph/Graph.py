import clip
import torch

import numpy as np

from PIL import Image
from lang_graph.Node import Node

class Graph:
    
    def __init__(self, model='ViT-B/32', device='cuda'):
        self.nodes = []
        self.images = []
        self.text = []

        self.device = device
        self.image_features = None
        self.text_features = None

        print(f"=========== Loading model {model} with device {device} ===========")
        self.model, self.preprocess = clip.load(model, device=device)

    # Add a node to the graph
    def add_node(self, node: Node):
        self.nodes.append(node)
        self.images.append(node.get_image())
        self.text.append(node.get_description())

    # Encode the images
    def encode_images(self):

        images = torch.stack([self.preprocess(image) for image in self.images]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)

        self.image_features = image_features
        return image_features
    
    # Encode the text
    def encode_text(self):

        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(self.text).to(self.device))

        self.text_features = text_features
        return text_features
    
    # Query the graph by text instruction
    def query_text(self, text: str):

        # Encode the text
        query_features = self.model.encode_text(clip.tokenize([text]).to(self.device))

        # Calculate the similarity
        similarity = (100 * query_features @ self.image_features.T)

        # Get the most similar node
        max_similarity, max_index = similarity[0].max(dim=0)

        return self.nodes[max_index.item()], max_similarity.item(), max_index.item()

# Test the graph
if __name__ == "__main__":

    # Log the start point
    print("============= Testing the graph =============")

    # Load the image
    img1 = Image.open("./data/img/bed.png")
    img2 = Image.open("./data/img/barbell.png")
    img3 = Image.open("./data/img/blue_chair.png")
    img4 = Image.open("./data/img/org_chair_table.png")

    # Create the nodes
    node1 = Node(img1, "Bed", np.array([0, 0]))
    node2 = Node(img2, "Barbell", np.array([0, 1]))
    node3 = Node(img3, "Blue chair", np.array([1, 0]))
    node4 = Node(img4, "Orange chair and table", np.array([1, 1]))

    # Create the graph
    graph = Graph()

    # Add the nodes to the graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    # Encode the images
    graph.encode_images()

    # Encode the text
    graph.encode_text()

    # Query the graph
    graph.query_text("Go to the bed")