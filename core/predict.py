import os
import sys
import glob
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig, TEST_DATA_DIR
from typing import Tuple, List
import numpy as np
import json
import cv2
import math
from pycocotools import mask as cocomask


class Predictor:
    config = MyMaskRcnnConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 30
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 320

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
        point_sets = []
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
            for i, res in enumerate(results):
                masks = res['masks']
                for i in range(masks.shape[-1]):
                    mask = masks[:, :, i]
                    segmentation = cocomask.encode(np.asfortranarray(mask, dtype=np.uint8))
                    score = 1
                    if len(res['scores'] > i):
                        score = res['scores'][i]
                    coco_id = res['coco_id']
                    point_sets.append((segmentation, score, coco_id))
            print("Contours extracted")
        return point_sets

    def predict_path(self, img_path: str, verbose=1) -> List[List[Tuple[int, int]]]:
        return self.predict_paths([img_path], verbose=verbose)

    def predict_paths(self, all_paths: List[str], verbose=1) -> List[List[Tuple[int, int]]]:
        all_images = []
        for p in all_paths:
            data = cv2.imread(p)
            coco_img_id = int(os.path.basename(p).replace(".jpg", ""))
            all_images.append((data, coco_img_id))
        return self.predict_arrays(images=all_images, verbose=verbose)


def test_images(annotations_file_name="predictions.json", nr_images=None, target_dir=TEST_DATA_DIR):
    predictor = Predictor(os.path.join(os.getcwd(), "model", "stage2.h5"))
    annotations_path = os.path.join(os.getcwd(), annotations_file_name)
    images = glob.glob(os.path.join(target_dir, "**/*.jpg"), recursive=True)
    if nr_images:
        # random.shuffle(images)
        images = images[:nr_images]

    point_sets_with_score = predictor.predict_paths(images, verbose=0)

    annotations = []
    count = 0
    print("Creating annotations")
    for segment, score, coco_img_id in point_sets_with_score:
        count += 1
        print("Creating annotation {}/{}".format(count, len(point_sets_with_score)))
        bbox = cocomask.toBbox(segment)
        seg = segment
        seg["counts"] = seg["counts"].decode('utf-8')
        ann = {
            "image_id": coco_img_id,
            "category_id": 100,
            "segmentation": seg,
            "bbox": bbox.tolist(),
            "score": float(score)
        }
        annotations.append(ann)
    with open(annotations_path, "w") as fp:
        json.dump(annotations, fp)


if __name__ == "__main__":
    nr_images = None
    path = TEST_DATA_DIR
    if len(sys.argv) > 1:
        nr_images = int(sys.argv[1])
    if len(sys.argv) > 2:
        path = sys.argv[2]
    print(sys.argv)

    test_images(nr_images=nr_images, target_dir=path)
    # test_images(nr_images=4)
    # test_images()
