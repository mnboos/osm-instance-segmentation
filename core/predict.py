import os
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig
from PIL import Image
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

    def predict(self, img_data):
        if not self._model:
            inference_config = self.InferenceConfig()
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        model = self._model
        res = model.detect([img_data], verbose=1)
        return res

    def predict_path(self, img_path):
        img = Image.open(img_path)
        data = np.asarray(img, dtype="uint8")
        return self.predict(data)


# if __name__ == "__main__":
#     p = Predictor(os.path.join(os.getcwd(), "model", "stage3_256px_overfitted.h5"))

