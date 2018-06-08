import os
import glob
from core.utils import is_filled
from core.mask_rcnn_config import TRAINING_DATA_DIR


def cleanup(folder):
    """
     * Removes masks which are nearly empty.
     * Then removes all images without any mask
    :param folder:
    :return:
    """
    print("Cleaning up: ", folder)
    masks = glob.glob(os.path.join(folder, "**/*.tif"), recursive=True)
    images = glob.glob(os.path.join(folder, "**/*.tiff"), recursive=True)
    for mask_path in masks:
        filled, fill_factor = is_filled(mask_path, 255, 0.004)
        if not filled:
            print("not filled: ", fill_factor, mask_path)
            os.remove(mask_path)
    for img_path in images:
        mask_path = img_path.replace(".tiff", "_*.tif")
        if not glob.glob(mask_path):
            print("Image without masks: ", img_path)
            os.remove(img_path)
    print("Cleanup finished")


if __name__ == "__main__":
    cleanup(TRAINING_DATA_DIR)
