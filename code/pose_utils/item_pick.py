import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator
from pose_utils.segment_inside_box import Segment_box



class Item_pick:
    """A class to manage person picking up items in a real-time video stream based on their poses."""

    def __init__(self):
        """Initializes the Item_pick with default values for Visual and Image parameters."""

        # Image and line thickness
        self.im0 = None
        self.tf = None

        # Keypoints and count information
        self.keypoints = None
        self.handraise_angle = None
        self.handnormal_angle = None
        self.threshold = 0.001

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.status = None
        self.pick_angle = "side_view"
        self.kpts_to_check = None

        # Visual Information
        self.view_img = False
        self.annotator = None
        self.segment_box = Segment_box()

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        kpts_to_check,
        line_thickness=2,
        view_img=False,
        handraise_angle=140.0,
        handnormal_angle=110.0,
        pose_type="top_view",
    ):
        
        #Configures the Item_pick line_thickness, save image and view image parameters.

        self.kpts_to_check = kpts_to_check
        self.tf = line_thickness
        self.view_img = view_img
        self.handraise_angle = handraise_angle
        self.handnormal_angle = handnormal_angle
        self.pose_type = pose_type
        self.line_thickness = line_thickness

    def plot_angle(self, angle, status,center_kpt):
        """
        Plot the angle and status of the pose.

        Args:
            angle (float): Angle of the pose
            status (str): Status of the pose
        """
        angle_text = f"Angle: {angle:.2f}"
        status_text = f"Status: {status}"
        font_scale = 0.6 + (self.line_thickness /10.0)

        #draw angle
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, font_scale, self.line_thickness)
        angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
        cv2.putText(self.im0, angle_text, angle_text_position, 0, font_scale, (0, 0, 255), self.line_thickness)

    def create_bbox_near_wrist(self, wrist_point,height=0,width=0):
        """
        Create a bounding box near the wrist point.

        Args:
            wrist_point (np.ndarray): Wrist point

        Returns:
            np.ndarray: Bounding box
        """
        bbox = [
            int(wrist_point[0] - 50),
            int(wrist_point[1] - 50),
            int(wrist_point[0] + 200),
            int(wrist_point[1] +  100),
        ]
        return bbox

    def verify_hand_area(self, index, k):
        """
        Verify the hand area of the pose.

        Args:
            index (int): Index of the keypoints to check
            k (np.ndarray): Keypoints
        """

        wrist_point = k[int(self.kpts_to_check[(index*3)+2])].cpu()
        wrist_point = np.array(wrist_point)
        #check non -zero
        if wrist_point[0] == 0 and wrist_point[1] == 0:
            return
        bbox = self.create_bbox_near_wrist(wrist_point)
        self.annotator.box_label( bbox, "Hand Area", color=(0, 255, 0))
        return self.segment_box.segment_inside_box(self.im0, bbox)
        print("Hand area verified")
        #segmentation inside the box


    def start_analysis(self, im0, results, frame_count):
        """
        Start counting the number of poses in the video frame.

        Args:
            im0 (np.ndarray): Image frame
            results (list): Results of pose estimation
            frame_count (int): Frame count of the video

        Returns:
            np.ndarray: Image frame with the count and stage information
        """
        self.im0 = im0
        if frame_count == 1:
            self.angle = [0] * int(len(self.kpts_to_check)/3)
            self.status = [0] * int(len(self.kpts_to_check)/3)
        self.keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=self.tf)

        num_keypoints = len(results[0])

        for k in reversed(self.keypoints):
            if self.pose_type in ["side_view", "top_view"]:
                for index,i in enumerate(range(0,len(self.kpts_to_check),3)):
                    

                    self.angle[index] = self.annotator.estimate_pose_angle(
                        k[int(self.kpts_to_check[i])].cpu(),
                        k[int(self.kpts_to_check[i+1])].cpu(),
                        k[int(self.kpts_to_check[i+2])].cpu(),
                    )
                    self.plot_angle(self.angle[index], self.status[0],center_kpt=k[int(self.kpts_to_check[i+1])])

                    self.im0 = self.annotator.draw_specific_points(
                        k,self.kpts_to_check,shape=(640,640) , radius=10)
                    
                    print(self.angle[1])
                    
                    if self.angle[1] < 90:
                        print("back")
                    if self.angle[index] > self.handraise_angle:
                        self.status[index] = "attempting_pickup"
                    elif self.angle[index] < self.handnormal_angle and self.status[index] == "attempting_pickup":
                        wrist = np.array(k[int(self.kpts_to_check[(index*3)+2])].cpu())
                        if wrist[0] != 0 and wrist[1] != 0:
                            self.status[index] = "maybe_picked"
                            #do segmentation here
                            return self.verify_hand_area(index,k)
                    
                    
            
                
        return self.im0
            
           
                






                


        