import os
from core.mask_rcnn_config import MyMaskRcnnConfig, OsmMappingDataset
from mask_rcnn.shapes import ShapesConfig, ShapesDataset
from mask_rcnn import model as modellib, utils

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn", "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

if not os.path.isfile(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = MyMaskRcnnConfig()
# config = ShapesConfig()
config.display()

# IMAGES_PATH = r"C:\Temp\images\training\split_small"
IMAGES_PATH = r"C:\Temp\images\training\split"
images = list(filter(lambda f: f.endswith(".tiff"), os.listdir(IMAGES_PATH)))

cutoffIndex = int(len(images)*.8)
trainingImages = images[0:cutoffIndex]
validationImages = images[cutoffIndex:-1]

# Training dataset
dataset_train = OsmMappingDataset(root_dir=IMAGES_PATH,
                                  img_width=config.IMAGE_SHAPE[0],
                                  img_height=config.IMAGE_SHAPE[1])
dataset_train.load(trainingImages)
dataset_train.prepare()

# Validation dataset
dataset_val = OsmMappingDataset(root_dir=IMAGES_PATH,
                                img_width=config.IMAGE_SHAPE[0],
                                img_height=config.IMAGE_SHAPE[1])
dataset_val.load(validationImages)
dataset_val.prepare()


# # Training dataset
# dataset_train = ShapesDataset()
# dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
# dataset_train.prepare()
#
# # Validation dataset
# dataset_val = ShapesDataset()
# dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
# dataset_val.prepare()


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

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 3+")
    model.train(train_dataset=dataset_train,
                val_dataset=dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='3+')

# Finetune layers from ResNet stage 3 and up
print("Training all")
model.train(train_dataset=dataset_train,
            val_dataset=dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=1000,
            layers='all')