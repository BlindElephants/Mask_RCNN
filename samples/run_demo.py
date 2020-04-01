import os, sys, random, math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)


file_names = next(os.walk(IMAGE_DIR))[2]
imagename = os.path.join(IMAGE_DIR, random.choice(file_names))
print("image: {}".format(imagename))
image = skimage.io.imread(imagename)
results = model.detect([image], verbose=0)
r = results[0]

def save_masks(masks, file_name):
    for i in range(masks.shape[2]):
        skimage.io.imsave("{}_mask_{}.png".format(file_name, i), img_as_uint(masks[:,:,i]))


save_masks(results[0]['masks'], "test_image")