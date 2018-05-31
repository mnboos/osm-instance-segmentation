import json
import os
import secrets
from airtiler import Airtiler
from core.settings import IMAGE_OUTPUT_FOLDER

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "core", "airtiler_config.json"), 'r') as f:
        config = json.load(f)
    config["options"]["target_dir"] = IMAGE_OUTPUT_FOLDER
    airtiler = Airtiler(bing_key=secrets.BING_KEY2)
    airtiler.process(config)
