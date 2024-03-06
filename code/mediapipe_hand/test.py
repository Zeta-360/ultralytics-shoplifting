import mediapipe as mp
import cv2
import os


mp.drawing = mp.solutions.drawing_utils
mp.hands = mp.solutions.hands
mp.holistic = mp.solutions.holistic

def create_hand_video(output_path,processed_frames):
    height, width, layers = processed_frames[0].shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join(output_path,"girl.mp4"), fourcc, 30.0, (width, height))

    for frame in processed_frames:
        out.write(frame)  # Write out frame to video

    # Release everything when job is finished
    out.release()
    cv2.destroyAllWindows()

def hand_pose_detection(video_path,output_path):
    processed_frames = []
    cap = cv2.VideoCapture(video_path)
    with mp.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.drawing.draw_landmarks(
                        image, hand_landmarks, mp.hands.HAND_CONNECTIONS)
            processed_frames.append(image)
            cv2.imshow('MediaPipe Hands', image)
           

            if cv2.waitKey(5) & 0xFF == 27:
                break

    #save video with hand landmarks using processed_frames
    processed_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames]


    
    out = cv2.VideoWriter(os.path.join(output_path,'output.mp4'), -1, 20.0, (640,480))
    if len(processed_frames) > 4:
        create_hand_video(output_path,processed_frames)
    cap.release()


if __name__ == "__main__":
    video_path = "../../data/vids/girl_lifting_4s.mp4"
    output_path = "../../outputs/mediapipe/vids"
    hand_pose_detection(video_path,output_path)
