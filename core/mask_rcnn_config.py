from mask_rcnn.config import Config
from mask_rcnn import utils
from core.training_data import get_instances
import os
import numpy as np
from PIL import Image

osm_class_ids = {
    'building': 1
}

IMAGE_WIDTH = 128


class MyMaskRcnnConfig(Config):
    NAME = "osm"

    NUM_CLASSES = 2  # building & not building

    # Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.001

    # 2 minutes
    STEPS_PER_EPOCH = 100 // IMAGES_PER_GPU

    # 1 hour epoch
    # STEPS_PER_EPOCH = 12000 // IMAGES_PER_GPU

    # Each tile is 256 pixels across, training data is 3x3 tiles
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128)
    # MASK_SHAPE = (IMAGE_MIN_DIM, IMAGE_MIN_DIM)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 64
    # DETECTION_MAX_INSTANCES = 64

    VALIDATION_STEPS = 100


class OsmMappingDataset(utils.Dataset):

    def __init__(self, root_dir, img_width, img_height):
        self.root_dir = root_dir
        self.img_width = img_width
        self.img_height = img_height
        utils.Dataset.__init__(self)

    def load(self, images):
        self.add_class("osm", 0, "building")
        print("")
        print("Loading {} images...".format(len(images)))
        for i in images:
            self.add_image(source="osm", image_id=i, path=os.path.join(self.root_dir, i))
        print("Loaded.")

    def load_image(self, image_id):
        info = self.image_info[image_id]
        #print("Load image: ", info["id"])
        image_path = os.path.join(self.root_dir, info["id"])
        img = Image.open(image_path)
        data = np.asarray(img, dtype="uint8")
        return data

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        #print("Load mask: ", info["id"])
        mask_path = os.path.join(self.root_dir, info["id"][:-1])  # images have fileextension ".tiff", masks have ".tif"

        instances = get_instances(mask_path)
        class_ids = np.zeros(len(instances), np.int32)

        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(instances)], dtype=np.uint8)
        for i, inst in enumerate(instances):
            class_ids[i] = osm_class_ids["building"]
            mask[:, :, i] = inst

        return mask, class_ids
