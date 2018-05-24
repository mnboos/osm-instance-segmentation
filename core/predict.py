import os
import sys
import glob
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig, TEST_DATA_DIR, IMAGE_WIDTH
from core.utils import get_contours
from typing import Tuple, List
import numpy as np
import json
import cv2
import math
from keras import backend as K


class Predictor:
    config = MyMaskRcnnConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        # IMAGES_PER_GPU = 30
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = 256
        IMAGE_MAX_DIM = 1024

    def __init__(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise RuntimeError("Weights cannot be found at: {}".format(weights_path))
        self.weights_path = weights_path
        self._model = None

    def predict_arrays(self, images: List[Tuple[np.ndarray, str]], verbose=1) -> List:
        batch_size = 30

        if not self._model:
            print("Loading model")
            inference_config = self.InferenceConfig()
            inference_config.BATCH_SIZE = batch_size
            inference_config.IMAGES_PER_GPU = batch_size
            print("Predicting {} images".format(len(images)))
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        model = self._model
        batches = int(math.ceil(len(images) / batch_size))
        all_point_sets = []
        for i in range(batches):
            start = i * batch_size
            end = start + batch_size
            img_with_id_batch = images[start:end]
            if len(img_with_id_batch) < batch_size:
                inference_config = self.InferenceConfig()
                inference_config.BATCH_SIZE = len(img_with_id_batch)
                inference_config.IMAGES_PER_GPU = len(img_with_id_batch)
                model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
                model.load_weights(self.weights_path, by_name=True)
            print("Predicting batch {}/{}".format(i, batches))
            img_batch = list(map(lambda i: i[0], img_with_id_batch))
            id_batch = list(map(lambda i: i[1], img_with_id_batch))
            results = model.detect(img_batch, image_ids=id_batch, verbose=verbose)
            print("Extracting contours...")
            for res in results:
                masks = res['masks']
                point_sets = get_contours(masks=masks)
                point_sets = list(map(lambda point_set: list(point_set), point_sets))
                image_id = res['coco_id']
                for points in point_sets:
                    all_point_sets.append((points, image_id))
            print("Contours extracted")
        K.clear_session()
        return all_point_sets

    def predict_path(self, img_path: str, verbose=1) -> List[List[Tuple[int, int]]]:
        return self.predict_paths([img_path], verbose=verbose)

    def predict_paths(self, all_paths: List[str], verbose=1) -> List[List[Tuple[int, int]]]:
        all_images = []
        for p in all_paths:
            data = cv2.imread(p)
            coco_img_id = int(os.path.basename(p).replace(".jpg", ""))
            all_images.append((data, coco_img_id))
        return self.predict_arrays(images=all_images, verbose=verbose)
