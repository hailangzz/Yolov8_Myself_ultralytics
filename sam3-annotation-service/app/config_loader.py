import yaml


def load_config(path="/app/config/config.yaml"):

    with open(path) as f:
        config = yaml.safe_load(f)

    return config
