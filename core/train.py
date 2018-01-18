import os
import random
from core.mask_rcnn_config import MyMaskRcnnConfig, OsmMappingDataset
from mask_rcnn import model as modellib, utils

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn", "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "images")
TRAINING_DATA_DIR = "/training-data"

if not os.path.isdir(TRAINING_DATA_DIR):
    windir = r"C:\Temp\images\training\output"
    if os.path.isdir(windir):
        TRAINING_DATA_DIR = windir
    else:
        raise RuntimeError("A directory '{}' is required containing the images to train the network"\
                           .format(TRAINING_DATA_DIR))

if not os.path.isfile(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# training_data.download(os.path.join(DATA_DIR, "raw"), TRAINING_DATA_DIR)

config = MyMaskRcnnConfig()
config.display()

print("Using training images in: ", TRAINING_DATA_DIR)

images = list(filter(lambda f: f.endswith(".tiff"), os.listdir(TRAINING_DATA_DIR)))
print("{} images found...".format(len(images)))
# random.shuffle(images)

cutoffIndex = int(len(images)*.8)
trainingImages = images[0:cutoffIndex]
validationImages = images[cutoffIndex:]

# Training dataset
dataset_train = OsmMappingDataset(root_dir=TRAINING_DATA_DIR,
                                  img_width=config.IMAGE_SHAPE[0],
                                  img_height=config.IMAGE_SHAPE[1])
dataset_train.load(trainingImages)
dataset_train.prepare()

# Validation dataset
dataset_val = OsmMappingDataset(root_dir=TRAINING_DATA_DIR,
                                img_width=config.IMAGE_SHAPE[0],
                                img_height=config.IMAGE_SHAPE[1])
dataset_val.load(validationImages)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    print(model.find_last()[1])
    model.load_weights(model.find_last()[1], by_name=True)

if init_with != "last":
    # Training - Stage 1
    # Adjust epochs and layers as needed
    print("Training network heads")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
    model.keras_model.save_weights(os.path.join(MODEL_DIR, "stage1.h5"), overwrite=True)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 3+")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='3+')
    model.keras_model.save_weights(os.path.join(MODEL_DIR, "stage2.h5"), overwrite=True)

# Finetune layers from ResNet stage 3 and up
print("Training all")
model.train(train_dataset=dataset_train,
            val_dataset=dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=1000,
            layers='all')
model.keras_model.save_weights(os.path.join(MODEL_DIR, "stage3.h5"), overwrite=True)
