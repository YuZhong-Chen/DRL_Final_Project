import argparse
import yaml # type: ignore

def _get_parser():

    parser = argparse.ArgumentParser(description="Training script for the model")

    # A list for text descriptions (a list of strings)
    parser.add_argument("--node_descriptions", type=list, default=["bed", "chair", "table"], help="List of text descriptions")

    # A list for input text
    parser.add_argument("--input_text", type=list, default=["Go to the sink with silver color"], help="List of input text")

    # Image directory
    parser.add_argument("--image_dir", type=str, default="./data/im", help="Directory containing images")

    # args = parser.parse_args()
    return parser

def _load_yaml(filename):
    with open(f"{filename}", "r") as file:
        config = yaml.safe_load(file)
    return config

def get_config(file="config/config.yaml"):

    # Get the parser and configuration file
    parser = _get_parser()

    # Load the configuration file from folder
    config = _load_yaml(file)

    # Set the default values to yaml configuration
    parser.set_defaults(**config)
    args = parser.parse_args(args = [])

    return args