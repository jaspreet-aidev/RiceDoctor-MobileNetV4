import os
import yaml

def load_config(config_name="config.yaml"):
    """
    Loads the configuration YAML file from the configs directory.
    """
    # Get the absolute path to the configs folder
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, config_name)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config