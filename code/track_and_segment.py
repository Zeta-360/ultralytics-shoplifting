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



track_history = defaultdict(lambda: [])
model = YOLO("yolov8x.pt")
names = model.model.names   
#current_dir = os.path.dirname(os.path.realpath(__file__))
Fast_SAM_frames = []

# Add the current directory to the system path
#sys.path.append(current_dir)
SAM_model= FastSAM(os.path.join("weights/FastSAM.pt"))

def stitch_video_FastSAM():
    #load frame
    frame = cv2.imread(os.path.join("outputs","FastSAM","outs","image0.jpg"))
    Fast_SAM_frames.append(frame)

def create_SAM_video(input_file_base_name):
    height, width, layers = Fast_SAM_frames[0].shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join('outputs/FastSAM',input_file_base_name), fourcc, 30.0, (width, height))

    for frame in Fast_SAM_frames:
        out.write(frame)  # Write out frame to video

    # Release everything when job is finished
    out.release()
    cv2.destroyAllWindows()


def segment_inside_box(frame, x, y, x2, y2):
    box = [x,y,x2,y2]
    float_box = [tensor.item() for tensor in box]
    DEVICE = 0
    everything_results = SAM_model(frame, device=DEVICE, retina_masks=True, conf=0.3, iou=0.9)
    prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
    masks = prompt_process.box_prompt(bbox = float_box)
    
    annotated_image = prompt_process.plot(annotations=masks, output=os.path.join(current_dir,"outs","video1_1"))

    # annotated_image=annotate_image(frame, masks=masks)
    # sv.plot_image(image=annotated_image, size=(8, 8))

def pose_estimation(frame):
    # Load the model
    pose_model = YOLO('yolov8n-pose.pt')
    results = pose_model(frame)[0]
    # Extract prediction results of pose estimation
    names = {0: "person"}
    for result in results.keypoints:
        pass

  
    

def main(input_path):
    input_file_base_name = os.path.basename(input_path)
    
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(os.path.join("outputs/FastSAM/outs",f"{input_file_base_name}_object_tracking.avi"),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            print("Frame: ", frame.shape)
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
                    #draws bounding box
                    frame_box = cv2.rectangle(frame.copy(), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cropped_frame = frame_box[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #### FastSAM for segmenting inside the bounding box####
                    segment_inside_box(frame, box[0], box[1], box[2], box[3])

                    stitch_video_FastSAM()
                    #### Pose Estimation ####
                    # pose_model = YOLO('yolov8n-pose.pt') 
                    # pose_estimation(frame)
                    
                    
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
            #cv2.imshow("Object Tracking", frame)

            result.write(frame)
        
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


    create_SAM_video(input_file_base_name)       
    result.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_file_path = "/media/i9/1TB_JAPAN/VSCODE/COOL_GITS/ultralytics-shoplifting/data/vids/1_1_crop.mp4"
    

    main(input_file_path)   
