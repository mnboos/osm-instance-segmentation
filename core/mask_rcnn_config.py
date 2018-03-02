from mask_rcnn.config import Config
from mask_rcnn import utils
from core.training_data import get_instances
from core.settings import IMAGE_WIDTH
from typing import Tuple
import os
import numpy as np
from PIL import Image

osm_class_ids = {
    'building': 1
}


class MyMaskRcnnConfig(Config):
    NAME = "osm"

    NUM_CLASSES = 2  # building & not building

    # Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    LEARNING_RATE = 0.001

    # faster training
    STEPS_PER_EPOCH = 100 // IMAGES_PER_GPU

    # all images
    # STEPS_PER_EPOCH = 100000 // IMAGES_PER_GPU

    # Each tile is 256 pixels across, training data is 3x3 tiles
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH

    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (128, 128)
    # MASK_SHAPE = (IMAGE_MIN_DIM, IMAGE_MIN_DIM)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 64
    # DETECTION_MAX_INSTANCES = 64

    VALIDATION_STEPS = 100


class OsmMappingDataset(utils.Dataset):

    def __init__(self):
        utils.Dataset.__init__(self)
        print("Dataset: OsmMappingDataset")

    def load(self, images):
        self.add_class("osm", 0, "building")
        print("")
        print("Loading {} images...".format(len(images)))
        for idx, path in enumerate(images):
            self.add_image(source="osm", image_id=idx, path=path)
        print("Loaded.")

    def _get_image(self, path: str) -> np.ndarray:
        # info = self.image_info[path]
        # image_path = info["path"]
        img = Image.open(path)
        data = np.asarray(img, dtype="uint8")
        return data

    def _get_mask(self, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # images have fileextension ".tiff", masks have ".tif"
        assert not mask_path.endswith(".tiff")
        if not os.path.isfile(mask_path):
            raise RuntimeError("Mask does not exist")

        instances = get_instances(mask_path)
        class_ids = np.zeros(len(instances), np.int32)

        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(instances)], dtype=np.uint8)
        for i, inst in enumerate(instances):
            class_ids[i] = osm_class_ids["building"]
            mask[:, :, i] = inst
        return mask, class_ids

    def load_image(self, image_id: str) -> np.ndarray:
        info = self.image_info[image_id]
        path = info["path"]
        return self._get_image(path)

    def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._get_mask(image_path[:-1])


class InMemoryDataset(OsmMappingDataset):
    def __init__(self):
        OsmMappingDataset.__init__(self)
        self._cache = {}
        print("Dataset: InMemoryDataset")

    def load(self, images):
        self.add_class("osm", 0, "building")
        print("")
        print("Loading {} images...".format(len(images)))
        for image_path in images:
            self.add_image(source="osm", image_id=image_path, path=image_path)
            self._cache[image_path] = {
                "img": self._get_image(path=image_path),
                "mask": self._get_mask(image_path[:-1])
            }
        print("Loaded.")

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._cache[image_path]["img"]

    def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._cache[image_path]["mask"]
