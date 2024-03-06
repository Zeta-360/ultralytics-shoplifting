from imports import *

from pose_utils.item_pick import Item_pick
#change current directory to the directory of the script
os.chdir(os.path.dirname(os.path.realpath(__file__)))


track_history = defaultdict(lambda: [])
pose_model = YOLO("yolov8x-pose.pt")

item_pick_object = Item_pick()
  



def pose_estimation(frame, frame_count):
    # Load the model
    results = pose_model.predict(frame, verbose=False)
    item_pick_object.set_args(kpts_to_check=[6, 8, 10 ,5,7,9], line_thickness=2, view_img=True, pose_type="side_view")
    pose_frame = item_pick_object.start_analysis(frame, results, frame_count)
    return pose_frame

 

def main(input_path):
    input_file_base_name = os.path.basename(input_path)[:-4]
    
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    uid = str(uuid.uuid1())[:8]
    #uid = uid[:8]
    result = cv2.VideoWriter(os.path.join("..","outputs/pose_detection/outs",f"{input_file_base_name}_pose{uid}.mp4"),
                        cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))Q
                        
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            print("Frame: ", frame.shape)
            frame_count+=1
            print("Frame Count: ", frame_count)
            pose_frame = pose_estimation(frame, frame_count)
            #cv2.imshow("Pose Estimation", pose_frame)
            #seg_frame = segment_inside_box(pose_frame)

            ####  Setting Classes to only 0 for detecting only Person ####
           

            #### FastSAM for segmenting inside the bounding box####
            #seg_frame = segment_inside_box(frame, box[0], box[1], box[2], box[3])

         
            cv2.imshow("Pose Estimation", pose_frame)
            result.write(pose_frame)
        
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


    #create_SAM_video(input_file_base_name)       
    result.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_file_path = "../data/vids/girl_lifting_4s.mp4"
    input_file_path = "../data/vids/man_picking.mp4"
    input_file_path = "/media/i9/1TB_JAPAN/VSCODE/DATASETS/shoplift_2022/Dataset/Shoplifting/Shoplifting_(17).mp4"
    

    main(input_file_path)   
