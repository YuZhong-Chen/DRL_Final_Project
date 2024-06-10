#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from PIL import Image

from lang_graph.Node import Node
from lang_graph.Graph import Graph
from lang_graph.utils.ArgParser import get_config

# Import int
from std_msgs.msg import Int32 

# Minimal publisher
# Set next node: ros2 topic pub /lang_node_callback std_msgs/msg/Int32 data:\ 1\
class LangNodePub(rclpy.node.Node):

    def __init__(self, graph, input_text):
        super().__init__('lang_node_pub')

        # Int publisher
        self.publisher_ = self.create_publisher(Int32, 'lang_node', 10)

        # Int subscriber
        self.subscription = self.create_subscription(Int32, 'lang_node_callback', self.listener_callback, 10)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        # Flag for publish the next node
        self.publish_next = False

        self.graph = graph
        self.input_text = input_text

        # Flag for publish the next node
        self.publish_next = False

        self.current_index = 0

    def timer_callback(self):

        if self.publish_next:
            self.publish_next = False
            self.current_index  = min(self.current_index + 1, len(self.input_text) - 1)
        
        # if self.publish_next:
        msg = Int32()

        # Query the graph with the input text
        node, _, index = self.graph.query_text(self.input_text[self.current_index])

        # Send the node point
        msg.data = index

        self.publisher_.publish(msg)

    def listener_callback(self, msg):

        if msg.data == 1:
            self.publish_next = True

def create_graph(p, image_folder):

    image_dir = image_folder
    
    print(f"=========== Creating graph with images from {image_dir} ===========")

    graph = Graph()

    # Load all of the images and text descriptions
    for desp in p.node_descriptions:
        img = Image.open(f"{image_dir}/{desp[0]}")
        node = Node(img, desp[1], np.array(desp[2]))
        graph.add_node(node)
        print(f"---\nAdding node with:")
        print(f"    1. Original config {desp}")
        print(f"    2. Node {node}")

    print("=========== Done creating graph ===========")

    return graph

class ROSParamServer(rclpy.node.Node):

    def __init__(self):
        super().__init__("ros_param_server")

        self.declare_parameter("config_file", "config/config.yaml")
        self.declare_parameter("image_folder", "data/img")

    def get_params(self):
        return {
            "config_file": self.get_parameter("config_file").value,
            "image_folder": self.get_parameter("image_folder").value
        }

def main(args=None):

    # Setup the ROS2 node
    rclpy.init(args=args)

    # Get ROS2 parameters
    ros_param_server = ROSParamServer()

    # Get the configuration
    config_file = ros_param_server.get_params()["config_file"]
    image_folder = ros_param_server.get_params()["image_folder"]

    print(f"Configuration file: {config_file}")

    p = get_config(config_file)

    # Create the graph
    graph = create_graph(p, image_folder)

    # Encode the images
    graph.encode_images()

    # Encode the text
    graph.encode_text()

    # Create the ROS2 node
    lang_node_pub = LangNodePub(graph, input_text=p.input_text)

    while rclpy.ok():
        rclpy.spin_once(lang_node_pub)

if __name__ == "__main__":
    main()