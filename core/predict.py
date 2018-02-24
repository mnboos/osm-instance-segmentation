import os
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig
from core.utils import MarchingSquares, georeference, rectangularize
from typing import Iterable, Tuple, List
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

    def __init__(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise RuntimeError("Weights cannot be found at: {}".format(weights_path))
        self.weights_path = weights_path
        self._model = None

    def predict_array(self, img_data: np.ndarray, tile: Tile = None) \
            -> List[List[Tuple[int, int]]]:
        if not self._model:
            print("Loading model")
            inference_config = self.InferenceConfig()
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        model = self._model
        res = model.detect([img_data], verbose=1)
        point_sets = self._get_buildings(masks=res[0]['masks'])
        rectangularized_outlines = []
        for p in point_sets:
            reg = rectangularize(p)
            rectangularized_outlines.append(reg)
        if tile:
            point_sets = map(lambda p: georeference(p, tile), rectangularized_outlines)
        return point_sets

    def predict_path(self, img_path: str, tile: Tile = None) -> List[List[Tuple[int, int]]]:
        img = Image.open(img_path)
        data = np.asarray(img, dtype="uint8")
        return self.predict_array(img_data=data, tile=tile)

    @staticmethod
    def _get_buildings(masks: np.ndarray) -> List[List[Tuple[int, int]]]:
        buildings: List[List[Tuple[int, int]]] = []
        for i in range(masks.shape[-1]):
            m = MarchingSquares.from_array(masks[:, :, i])
            points: List[Tuple[int, int]] = m.find_contour()
            buildings.append(points)
        return buildings
