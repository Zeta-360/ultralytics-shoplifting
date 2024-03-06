import cv2
from ultralytics import YOLO
import time
import os

# Load the YOLOv8 model
model = YOLO('yolov8l-world.pt')

# Open the video file
video_path = '../../data/vids/man_picking.mp4'
# video_path = '../../data/vids/girl_lifting_4s.mp4'
video_name = video_path.split('/')[-1].split('.')[0]

video_capture = cv2.VideoCapture(video_path)

final_classes = [""]
final_classes.extend(["person holding objects"])
final_classes.extend(["bag", "shopping cart"])
print(final_classes)
model.set_classes(final_classes)

if video_capture: # video
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) # integer required, floats produce error in MP4 codec
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
else: # stream
    fps, w, h = 30, 640, 480


output_base_dir = '../../data/yolo_world/'
epoch_time = int(time.time())
output_dir = os.path.join(output_base_dir, f'{video_name}_{epoch_time}')
output_dir_frames = os.path.join(output_dir, 'frames')
output_video_save_path = os.path.join(output_dir, f'{video_name}.mp4')

# os.mkdir(output_dir)
# os.mkdir(output_dir_frames)

video_writer = cv2.VideoWriter(output_video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

counter = 0
# Loop through the video frames
while video_capture.isOpened():
    # Read a frame from the video
    success, frame = video_capture.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame
        # results[0].save(os.path.join(output_dir_frames, f'{counter}.jpg'))
        # counter += 1

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Write the frame to the output video
        video_writer.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video_capture.release()
# video_writer
cv2.destroyAllWindows()