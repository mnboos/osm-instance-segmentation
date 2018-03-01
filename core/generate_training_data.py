import json
import os
import secrets
from airtiler import Airtiler

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "core", "airtiler_config.json"), 'r') as f:
        config = json.load(f)
    airtiler = Airtiler(secrets.BING_KEY)
    airtiler.process(config)
