import json
import os
import secrets
from airtiler import Airtiler
from core.mask_rcnn_config import TRAINING_DATA_DIR

if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "core", "airtiler_config.json"), 'r') as f:
        config = json.load(f)
    config["options"]["target_dir"] = TRAINING_DATA_DIR
    airtiler = Airtiler(bing_key=secrets.BING_KEY2)
    airtiler.process(config)
