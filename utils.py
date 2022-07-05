import json


def load_config():
    with open("config/config.json") as json_file:
        config_file = json.load(json_file)
    return config_file