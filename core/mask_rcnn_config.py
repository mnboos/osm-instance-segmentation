from mask_rcnn.config import Config
from mask_rcnn import utils
from core.training_data import get_instances, get_instances_from_array
from core.settings import IMAGE_WIDTH
from core.utils import osm_class_ids
from typing import Tuple
import os
import glob
import numpy as np
from PIL import Image

TRAINING_DATA_DIR = "/training-data"
TEST_DATA_DIR = "/test-data"

if not os.path.isdir(TRAINING_DATA_DIR):
    TRAINING_DATA_DIR = r"D:/training_images/_new"
    TEST_DATA_DIR = r"D:/training_images/_new"


class MyMaskRcnnConfig(Config):
    NAME = "osm"

    NUM_CLASSES = len(osm_class_ids)+1  # classes & not any of those

    # Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001

    # faster training
    STEPS_PER_EPOCH = 8800 // IMAGES_PER_GPU
    # STEPS_PER_EPOCH = 280741 // IMAGES_PER_GPU

    # all images
    # STEPS_PER_EPOCH = 1000

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

    VALIDATION_STEPS = 100  # 60317

    MEAN_PIXEL = np.array([63.1, 81.3, 83.1])


class OsmMappingDataset(utils.Dataset):

    def __init__(self, images):
        utils.Dataset.__init__(self)
        self.images = images
        print("Dataset: OsmMappingDataset")

    def load(self):
        images = self.images
        for class_name in osm_class_ids:
            self.add_class("osm", osm_class_ids[class_name], class_name)
        print("")
        print("Loading {} images...".format(len(images)))
        for idx, path in enumerate(images):
            self.add_image(source="osm", image_id=idx, path=path)
        print("Loaded.")

    @staticmethod
    def _get_image(path: str) -> np.ndarray:
        # info = self.image_info[path]
        # image_path = info["path"]
        img = Image.open(path)
        data = np.asarray(img, dtype="uint8")
        return data

    @staticmethod
    def _get_mask(mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # images have fileextension ".tiff", masks have ".tif"
        assert not mask_path.endswith(".tiff")

        all_instances = []
        all_mask_paths = glob.glob(mask_path.replace(".tif", "_*.tif"))
        nr_instances = 0
        for p in all_mask_paths:
            current_instances = get_instances(p)
            class_name = p.replace(".tif", "").split("_")[-1]
            all_instances.append((class_name, current_instances))
            nr_instances += len(current_instances)

        class_ids = np.zeros(nr_instances, np.int32)

        count = 0
        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, nr_instances], dtype=np.uint8)
        for class_name, current_instances in all_instances:
            for inst in current_instances:
                class_ids[count] = osm_class_ids[class_name]
                mask[:, :, count] = inst
                count += 1
        return mask, class_ids

    def load_image(self, image_id: str) -> np.ndarray:
        info = self.image_info[image_id]
        path = info["path"]
        return self._get_image(path)

    def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._get_mask(image_path[:-1])

#
# class CocoDataset(utils.Dataset):
#
#     def __init__(self, path, limit=None, annotation_filename="annotation.json"):
#         utils.Dataset.__init__(self)
#         self._cache = {}
#         print("Dataset: InMemoryDataset")
#         self.no_logging = False
#         self.path = path
#         self.coco = COCO(os.path.join(path, annotation_filename))
#         self.limit = limit
#         self.coco_images = None
#
#     def load(self):
#         image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
#         images = self.coco.loadImgs(image_ids)
#         self.coco_images = images
#
#         for class_name in osm_class_ids:
#             self.add_class("osm", osm_class_ids[class_name], class_name)
#         print("")
#         print("Loading {} images...".format(len(images)))
#         progress = 0
#         total_nr_images = len(images)
#         for idx, coco_img in enumerate(images):
#             if self.limit and idx >= self.limit:
#                 break
#
#             image_path = os.path.join(self.path, "images", coco_img["file_name"])
#             self.add_image(source="osm", image_id=image_path, path=image_path, coco_id=coco_img["id"])
#
#             new_progress = int(round(idx / total_nr_images * 100))
#             if not self.no_logging and new_progress != progress:
#                 progress = new_progress
#                 print("Caching progress: {}% ({} images)".format(progress, idx+1))
#                 sys.stdout.flush()
#         print("Loaded.")
#
#     def _get_image(self, path: str) -> np.ndarray:
#         data = cv2.imread(path)
#         return data
#
#     def load_image(self, image_id: str) -> np.ndarray:
#         info = self.image_info[image_id]
#         path = info["path"]
#         return self._get_image(path)
#
#     def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
#         info = self.image_info[image_id]
#         coco_id = info["coco_id"]
#         coco_img = list(filter(lambda i: i["id"] == coco_id, self.coco_images))[0]
#         return self.get_mask_from_annotation(coco_img)
#
#     def get_mask_from_annotation(self, img):
#         annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
#         annotations = self.coco.loadAnns(annotation_ids)
#         # all_instances = np.zeros((img['height'], img['width']), dtype=np.uint8)
#         # print("nr annotations: ", len(annotations))
#
#         annotations = list(filter(lambda a: a["area"] >= 100, annotations))
#
#         class_ids = np.zeros(len(annotations), np.int32)
#         # print("Nr instances:", len(annotations))
#         mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(annotations)], dtype=np.uint8)
#
#         for i, ann in enumerate(annotations):
#             rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
#             m = cocomask.decode(rle)
#             m = m.reshape((img['height'], img['width']))
#             mask[:, :, i] = m
#             class_ids[i] = osm_class_ids["building"]
#         return mask, class_ids
#
#     # @staticmethod
#     # def get_mask_from_array(arr) -> Tuple[np.ndarray, np.ndarray]:
#     #     instances = get_instances_from_array(arr)
#     #     class_ids = np.zeros(len(instances), np.int32)
#     #
#     #     print("Nr instances:", len(instances))
#     #     mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(instances)], dtype=np.uint8)
#     #     for i, inst in enumerate(instances):
#     #         class_ids[i] = osm_class_ids["building"]
#     #         mask[:, :, i] = inst
#     #     return mask, class_ids
#     #
#     # def get_mask_from_annotation(self, img):
#     #     annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
#     #     annotations = self.coco.loadAnns(annotation_ids)
#     #     # if len(annotations) == 4:
#     #     #     print(annotations)
#     #     all_instances = np.zeros((img['height'], img['width']), dtype=np.uint8)
#     #     print("nr annotations: ", len(annotations))
#     #     for ann in annotations:
#     #         if ann["area"] >= 100:
#     #             rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
#     #             m = cocomask.decode(rle)
#     #             m = m.reshape((img['height'], img['width']))
#     #             # print("max: ", m.max())
#     #             all_instances[np.where(m >= 1)] = 255
#     #     return self.get_mask_from_array(all_instances)
#
#
# class CocoInMemoryDataset(CocoDataset):
#     def __init__(self, path, limit=None, annotation_filename="annotation.json"):
#         CocoDataset.__init__(self, path=path, limit=limit, annotation_filename=annotation_filename)
#         # self._cache = {}
#         # print("Dataset: InMemoryDataset")
#         # self.no_logging = False
#         # self.path = path
#         # self.coco = COCO(os.path.join(path, annotation_filename))
#         # self.limit = limit
#
#     def load(self):
#         image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
#         images = self.coco.loadImgs(image_ids)
#
#         for class_name in osm_class_ids:
#             self.add_class("osm", osm_class_ids[class_name], class_name)
#         print("")
#         print("Loading {} images...".format(len(images)))
#         progress = 0
#         total_nr_images = len(images)
#         for idx, coco_img in enumerate(images):
#             if self.limit and idx >= self.limit:
#                 break
#
#             image_path = os.path.join(self.path, "images", coco_img["file_name"])
#             img = self._get_image(path=image_path)
#             msk = self.get_mask_from_annotation(coco_img)
#             self.add_image(source="osm", image_id=image_path, path=image_path)
#             self._cache[image_path] = {
#                 "img": img,
#                 "mask": msk
#             }
#
#             new_progress = int(round(idx / total_nr_images * 100))
#             if not self.no_logging and new_progress != progress:
#                 progress = new_progress
#                 print("Caching progress: {}% ({} images)".format(progress, idx+1))
#                 sys.stdout.flush()
#         print("Loaded.")
#
#     def load_image(self, image_id):
#         info = self.image_info[image_id]
#         image_path = info["path"]
#         return self._cache[image_path]["img"]
#
#     def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
#         info = self.image_info[image_id]
#         image_path = info["path"]
#         return self._cache[image_path]["mask"]
