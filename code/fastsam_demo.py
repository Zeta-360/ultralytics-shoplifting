from imports import *
from pose_utils.item_pick import Item_pick
#change current directory to the directory of the script
os.chdir(os.path.dirname(os.path.realpath(__file__)))


track_history = defaultdict(lambda: [])
pose_model = YOLO("yolov8x-pose.pt")

item_pick_object = Item_pick()
  
Fast_SAM_frames = []

SAM_model= FastSAM(os.path.join("../weights/FastSAM.pt"))


def pose_estimation(frame):
    # Load the model
    results = pose_model.predict(frame, verbose=False)
    item_pick_object.set_args(kpts_to_check=[6, 8, 10 ,5,7,9], line_thickness=2, view_img=True, pose_type="side_view")
    pose_frame = item_pick_object.start_analysis(frame, results, 1)
    return pose_frame

def segment_inside_box(frame):
    # box = [x,y,x2,y2]
    # float_box = [tensor.item() for tensor in box]
    DEVICE = 0
    everything_results = SAM_model(frame, device=DEVICE, retina_masks=True, conf=0.3, iou=0.9)
    prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
    masks = prompt_process.everything_prompt()
    #masks = prompt_process.box_prompt(bbox = float_box)
    
    
    prompt_process.plot(annotations=masks, output=os.path.join("..","outputs/FastSAM","outs"))
    #read saved out image using cv2
    annotated_image = cv2.imread(os.path.join("..","outputs/FastSAM","outs","image0.jpg"))
    return annotated_image
    # annotated_image=annotate_image(frame, masks=masks)
    # sv.plot_image(image=annotated_image, size=(8, 8))




 

  
    

def main(input_path,output_folder_path):
    input_file_base_name = os.path.basename(input_path)[:-4]
    
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    uid = str(uuid.uuid1())[:8]
    #uid = uid[:8]
    result = cv2.VideoWriter(os.path.join(output_folder_path,f"{input_file_base_name}_samonly{uid}.mp4"),
                        cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
                        

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            print("Frame: ", frame.shape)
            

           
            seg_frame = segment_inside_box(frame)
            cv2.imshow("Segmentation", seg_frame)

         
  
            result.write(seg_frame)
        
            
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
    
    output_folder_path = "../outputs/FastSAM/outs"
    main(input_file_path,output_folder_path)   
