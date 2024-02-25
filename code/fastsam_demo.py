from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import supervision as sv
import sys
# Get the current file directory

import torch

def annotate_image(image_path: str, masks: np.ndarray) -> np.ndarray:
    image = cv2.imread(image_path)

    xyxy = sv.mask_to_xyxy(masks=masks)
    detections = sv.Detections(xyxy=xyxy, mask=masks)

    mask_annotator = sv.MaskAnnotator()
    return mask_annotator.annotate(scene=image.copy(), detections=detections)

current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the current directory to the system path
sys.path.append(current_dir)
model = FastSAM(os.path.join(current_dir,"weights/FastSAM.pt"))

# Load the image
# IMAGE_PATH = os.path.join(current_dir,"resources/people_7.jpg")
# IMAGE_PATH = "/opt/homebrew/runs/detect/predict3/girl_lifting_4s_frames/1.jpg"
IMAGE_PATH = "/opt/homebrew/runs/detect/predict4/man_picking_frames/2.jpg"
try:
    image = cv2.imread(os.path.join(current_dir, IMAGE_PATH))
except:
    image = cv2.imread(os.path.join(IMAGE_PATH))
# image = cv2.imread(os.path.join(current_dir,"resources/people_7.jpg"))

# DEVICE = 0
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=640, conf=0.1, iou=0.5,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

masks = prompt_process.box_prompt(bbox=[0, 0, 1920, 1080])
prompt_process.plot(annotations=masks, output=os.path.join(current_dir,"outs"))


# masks = masks.numpy().astype(bool)
# annotated_image=annotate_image(image_path=IMAGE_PATH, masks=masks)
print("Segmentation complete")



# Segment the image
