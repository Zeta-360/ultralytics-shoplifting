import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
import sys
import os
import torch
import roboflow
import base64

from roboflow import Roboflow
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

import torch

track_history = defaultdict(lambda: [])
model = YOLO("yolov8x.pt")
names = model.model.names   
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the current directory to the system path
sys.path.append(current_dir)
SAM_model= FastSAM(os.path.join(current_dir,"weights/FastSAM.pt"))

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def segment_inside_box(frame, x, y, x2, y2):
    # DEVICE = 0
    everything_results = SAM_model(frame, device=DEVICE, retina_masks=True, imgsz=416, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)

    masks = prompt_process.box_prompt(bbox=[0, 0, 1800, 3000])
    prompt_process.plot(annotations=masks, output=os.path.join(current_dir,"outs"))



video_path = "../girl_lifting_4s.mp4"
# video_path = "resources/1_3_crop.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("object_tracking.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        ####  Setting Classes to only 0 for detecting only Person ####
        results = model.track(frame, persist=True, verbose=False, classes=[0])
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                #### FastSAM for segmenting inside the bounding box####
                segment_inside_box(frame, box[0], box[1], box[2], box[3])
                
                
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        #view output live
        cv2.imshow("Object Tracking", frame)

        # result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()