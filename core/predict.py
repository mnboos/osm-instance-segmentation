import os
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig
from core.utils import MarchingSquares, georeference
from typing import Iterable, Tuple
from PIL import Image
from pygeotile.tile import Tile
import numpy as np


class Predictor:
    config = MyMaskRcnnConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = 256
        IMAGE_MAX_DIM = 256

    def __init__(self, weights_path):
        if not os.path.isfile(weights_path):
            raise RuntimeError("Weights cannot be found at: {}".format(weights_path))
        self.weights_path = weights_path
        self._model = None

    def predict(self, img_data: np.ndarray, tile: Tile = None) -> Iterable[Tuple[int, int]]:
        if not self._model:
            inference_config = self.InferenceConfig()
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        model = self._model
        res = model.detect([img_data], verbose=1)
        point_sets = self._get_buildings(masks=res[0]['masks'])
        if tile:
            point_sets = map(lambda p: georeference(p, tile), point_sets)
        return point_sets

    @staticmethod
    def _get_buildings(masks: np.ndarray) -> Iterable[Tuple[int, int]]:
        buildings = []
        for i in range(masks.shape[-1]):
            m = MarchingSquares.from_array(masks[:, :, i])
            points = m.find_contour()
            buildings.append(points)
        return buildings

    def predict_path(self, img_path: str, tile: Tile = None) -> Iterable[Tuple[int, int]]:
        img = Image.open(img_path)
        data = np.asarray(img, dtype="uint8")
        return self.predict(img_data=data, tile=tile)
