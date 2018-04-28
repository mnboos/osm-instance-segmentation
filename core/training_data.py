from PIL import Image
import shutil
import sys
from typing import List
import os
import urllib.request
import datetime
import math
import numpy as np
from scipy import ndimage
from core.settings import IMAGE_WIDTH
import cv2


class FileTypes:
    IMAGE = "sat"
    MASK = "map"


test_data = [
    {
        "url": "http://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/{type}//{filename}",
        "images": ["22678915_15.tif", "22678930_15.tif", "22678945_15.tif", "22678960_15.tif", "22678975_15.tif", "22678990_15.tif", "22679005_15.tif", "22679020_15.tif", "22679035_15.tif", "22679050_15.tif", "22828915_15.tif", "22828945_15.tif", "22828960_15.tif", "22828975_15.tif", "22829005_15.tif", "22829020_15.tif", "22829035_15.tif", "22978870_15.tif", "22978885_15.tif", "22978900_15.tif", "22978915_15.tif", "22978930_15.tif", "22978960_15.tif", "22978975_15.tif", "22978990_15.tif", "22979005_15.tif", "22979020_15.tif", "22979035_15.tif", "22979050_15.tif", "22979065_15.tif", "23128870_15.tif", "23128885_15.tif", "23128900_15.tif", "23128915_15.tif", "23128930_15.tif", "23128945_15.tif", "23128960_15.tif", "23128975_15.tif", "23128990_15.tif", "23129005_15.tif", "23129020_15.tif", "23129035_15.tif", "23129050_15.tif", "23129065_15.tif", "23129125_15.tif", "23129140_15.tif", "23129155_15.tif", "23129170_15.tif", "23278885_15.tif", "23278900_15.tif", "23278915_15.tif", "23278930_15.tif", "23278945_15.tif", "23278960_15.tif", "23278975_15.tif", "23278990_15.tif", "23279005_15.tif", "23279020_15.tif", "23279035_15.tif", "23279050_15.tif", "23279080_15.tif", "23279095_15.tif", "23279140_15.tif", "23279155_15.tif", "23279170_15.tif", "23428900_15.tif", "23428915_15.tif", "23428930_15.tif", "23428945_15.tif", "23428960_15.tif", "23428975_15.tif", "23428990_15.tif", "23429005_15.tif", "23429035_15.tif", "23429050_15.tif", "23429065_15.tif", "23429095_15.tif", "23429125_15.tif", "23429140_15.tif", "23429170_15.tif", "23578915_15.tif", "23578930_15.tif", "23578945_15.tif", "23578975_15.tif", "23578990_15.tif", "23579020_15.tif", "23579035_15.tif", "23579065_15.tif", "23579080_15.tif", "23579095_15.tif", "23579110_15.tif", "23579125_15.tif", "23579140_15.tif", "23728840_15.tif", "23728945_15.tif", "23728960_15.tif", "23728975_15.tif", "23728990_15.tif", "23729005_15.tif", "23729020_15.tif", "23729050_15.tif", "23729065_15.tif", "23729080_15.tif", "23729095_15.tif", "23729110_15.tif", "23878915_15.tif", "23878930_15.tif", "23878945_15.tif", "23878975_15.tif", "23878990_15.tif", "23879020_15.tif", "23879035_15.tif", "23879050_15.tif", "23879065_15.tif", "23879095_15.tif", "23879110_15.tif", "24029035_15.tif", "24029050_15.tif", "24029065_15.tif", "24029080_15.tif", "24029110_15.tif", "24179020_15.tif", "24179035_15.tif", "24179050_15.tif", "24179080_15.tif", "24328840_15.tif", "24328855_15.tif", "24328870_15.tif", "24329020_15.tif", "24329035_15.tif", "24329095_15.tif", "24478840_15.tif", "24478855_15.tif", "24478870_15.tif", "24478885_15.tif", "24478900_15.tif", "24479005_15.tif"]
    },
    {
        "url": "http://www.cs.toronto.edu/~vmnih/data/mass_buildings/test/{type}//{filename}",
        "images": ["22828930_15.tif", "22828990_15.tif", "22829050_15.tif", "23429020_15.tif", "23429080_15.tif", "23578960_15.tif", "23579005_15.tif", "23729035_15.tif", "23879080_15.tif", "24179065_15.tif"]
    },
    {
        "url": "http://www.cs.toronto.edu/~vmnih/data/mass_buildings/valid/{type}//{filename}",
        "images": ["22978945_15.tif", "23429155_15.tif", "23579050_15.tif", "23728930_15.tif"]
    }]


def download(download_folder, tiled_output_folder, tile_size=750):
    print("Starting download...")
    for test_data_obj in test_data:
        all_downloaded=False
        while not all_downloaded:
            all_downloaded = True
            for index, file_name in enumerate(test_data_obj["images"]):
                for is_image in range(0, 2):
                    if is_image:
                        file_name += 'f'
                        url = test_data_obj["url"].format(type=FileTypes.IMAGE, filename=file_name)
                    else:
                        url = test_data_obj["url"].format(type=FileTypes.MASK, filename=file_name)
                    target_path = os.path.join(download_folder, file_name)
                    if not os.path.isfile(target_path):
                        all_downloaded = False
                        what = "mask"
                        if is_image:
                            what = "image"
                        print("Downloading {what} {nr} {source}".format(what=what, nr=index, source=url))
                        try:
                            urllib.request.urlretrieve(url, target_path)
                        except:
                            print("!i!i!i!i========>>>>> Download failed")
    print("Download complete. Tiles will now be created")

    create_tiles(download_folder, tiled_output_folder, tile_size=tile_size)

    print("Finished: ", datetime.datetime.utcnow())


def is_uni(img, color, fill_factor, img_width, img_height, convert=None):
    if convert:
        clrs = list(img.convert(convert).getcolors())
    else:
        clrs = list(img.getcolors())
    clrs = sorted(clrs, key=lambda v: v[0]*-1)
    color_count = clrs[0][0]
    is_uni = clrs[0][1] == color and color_count >= (img_width*img_height) * fill_factor
    # print(clrs)
    # if is_uni:
    #     print(clrs)
    return is_uni


def create_tiles(source_folder, target_folder, tile_size, limit=None):
    """
     * Generates the files from the images in the source folder. A target folder will be created with the name of
       the tile_size.
    :param source_folder:
    :param target_folder:
    :param tile_size:
    :return:
    """
    # target_folder = os.path.join(target_folder, str(tile_size))
    if not os.path.isdir(target_folder):
        shutil.os.makedirs(target_folder)
    files = os.listdir(source_folder)
    images = list(filter(lambda f: f.endswith(".tif"), files))
    print("Tiling images...")
    progress = "0.00%"
    count = 0
    for index, i in enumerate(images):
        progress_new = "{0:.0f}%".format(index / len(images) * 100)
        if progress != progress_new:
            print("{} - ({})".format(progress_new, datetime.datetime.utcnow()))
            progress = progress_new
        mask = Image.open(os.path.join(source_folder, i))
        img = Image.open(os.path.join(source_folder, i + 'f'))
        nr_horiz = int(math.ceil(img.width / tile_size))
        nr_vert = int(math.ceil(img.height / tile_size))
        for ix in range(0, nr_horiz):
            if limit and count >= limit:
                break
            for iy in range(0, nr_vert):
                output_file = i.split('.')[0] + "_{x}_{y}.tif".format(x=ix, y=iy)
                target_img = os.path.join(target_folder, output_file + 'f')
                target_mask = os.path.join(target_folder, output_file)
                if not os.path.isfile(target_img) or not os.path.isfile(target_mask):
                    box = (ix * tile_size, iy * tile_size, ix * tile_size + tile_size, iy * tile_size + tile_size)
                    mask_cropped = mask.crop(box)
                    img_cropped = img.crop(box)
                    # make to grayscale and get colors
                    img_is_uni = is_uni(img=img_cropped, color=255, fill_factor=.25, img_width=tile_size, img_height=tile_size, convert="L")
                    mask_is_uni = is_uni(img=mask_cropped, color=(0,0,0), fill_factor=.85, img_width=tile_size, img_height=tile_size)
                    if img_is_uni or mask_is_uni:  # skip images where at least 25% of the image is white
                        continue

                    if limit and count >= limit:
                        break
                    count += 1
                    img_cropped.save(target_img)
                    mask_cropped.save(target_mask)


# def augment(directory):
#     files = os.listdir(directory)
#     p = Augmentor.Pipeline(source_directory=directory)
#     p.add_operation()
#     for f in files:
#         path = os.path.join(directory, f)

def get_instances(mask_path: str) -> List[np.ndarray]:
    # img = Image.open(mask_path)
    # imgray = img.convert('L')
    # data = np.asarray(imgray)
    data = cv2.imread(mask_path, 0)
    return get_instances_from_array(data)


def get_instances_from_array(data) -> List[np.ndarray]:
    labeled, num_features = ndimage.label(data)
    instances = []
    for i in range(1, num_features+1):
        x = np.zeros_like(data)
        x[labeled == i] = 1
        instances.append(x)
    return instances


source_folder = r"C:\Temp\images\training\raw"
if not os.path.isdir(source_folder):
    source_folder = "/source-images"

target_folder = r"C:\Temp\images\training\output"
if not os.path.isdir(target_folder):
    target_folder = "/training-data"

# instances = get_instances(os.path.join(r"C:\Temp\images", "test2.tif"))
# print(instances)


if __name__ == "__main__":
    create_tiles(source_folder=source_folder,
                 target_folder=target_folder,
                 tile_size=IMAGE_WIDTH,
                 limit=None)
# augment(r"C:\Temp\images\training")

