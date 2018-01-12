from mask_rcnn.config import Config
from mask_rcnn import utils
import os
import numpy as np
from PIL import Image

osm_class_ids = {
    'building': 1
}


class MyMaskRcnnConfig(Config):
    NAME = "OSM building mapping"

    NUM_CLASSES = 2  # building & not building

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (56, 56)

    MAX_GT_INSTANCES = 20

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    IMAGES_PER_GPU = 8

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


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
            self.add_image(source="osm", image_id=i, path=os.path.join(self.root_dir, i), width=128, height=128)
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

        class_ids = np.zeros(1, np.int32)
        class_ids[0] = osm_class_ids["building"]

        img = Image.open(mask_path)
        data = np.asarray(img, dtype="uint8")
        return data, class_ids
