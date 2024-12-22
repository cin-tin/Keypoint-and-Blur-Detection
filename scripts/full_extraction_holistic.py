import numpy as np
import mediapipe as mp
import os
import cv2

mp_holistic = mp.solutions.holistic  # Used to extract full-body keypoints
features_old = np.zeros((65, 3))

# Function for keypoint detection
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

# Function to extract keypoints
def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return True, face, pose, lh, rh

# Function to extract useful features from landmarks
def extract_features(face, pose, left_hand, right_hand, frame_width, frame_height):
    features = []
    # pose_filter = [11, 12, 13, 14, 15, 16]  # Shoulder, elbow, and wrist points
    # face_filter = [4, 13, 17, 61, 68, 93, 151, 175, 207, 291, 298, 427, 454, 33, 133, 362, 263]
    face_filter = [4, 13, 68, 93, 151, 175, 207, 298, 427, 454, 33, 133, 362, 263]
    pose_filter = [11, 12, 13, 14, 15, 16, 23, 24]
    lh_filter = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
    rh_filter = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]

    # Extract filtered face keypoints
    for indx, lm in enumerate(face):
        if indx in face_filter:
            x = lm[0] * frame_width
            y = lm[1] * frame_height
            features.append([x, y])

    # Extract filtered pose keypoints
    for indx, lm in enumerate(pose):
        if indx in pose_filter:
            x = lm[0] * frame_width
            y = lm[1] * frame_height
            features.append([x, y])

    # Extract filtered left hand keypoints
    for indx, lm in enumerate(left_hand):
        if indx in lh_filter:
            x = lm[0] * frame_width
            y = lm[1] * frame_height
            features.append([x, y])

    # Extract filtered right hand keypoints
    for indx, lm in enumerate(right_hand):
        if indx in rh_filter:
            x = lm[0] * frame_width
            y = lm[1] * frame_height
            features.append([x, y])

    return np.array(features)
    
    

    # for lm in left_hand:
    #     x = lm[0] * frame_width
    #     y = lm[1] * frame_height
    #     features.append([x, y])
    # for lm in right_hand:
    #     x = lm[0] * frame_width
    #     y = lm[1] * frame_height
    #     features.append([x, y])
    # return np.array(features)

# Define paths
dataset_dir = "/sample"
save_directory = "/full_extraction_hands/key_frame_nc-1"
frame_output_folder = "/full_extraction_hands/frames_nc-1"

# Create subdirectories for each category
categories = ["non_blur", "blur"]
for category in categories:
    category_path = os.path.join(frame_output_folder, category)
    os.makedirs(category_path, exist_ok=True)

# Processing videos
for video_file in os.listdir(dataset_dir):
    video_extensions = ['.mp4', '.mov', '.MOV', '.MP4']
    if not any(video_file.endswith(ext) for ext in video_extensions):
        continue

    video_path = os.path.join(dataset_dir, video_file)
    save_video_path = os.path.join(save_directory, video_file[:-4])
    os.makedirs(save_video_path, exist_ok=True)

    # Text file to store keypoints for this video
    keypoints_file = os.path.join(save_video_path, f"{video_file[:-4]}_keypoints.txt")
    with open(keypoints_file, "w") as kp_file:
        kp_file.write("Frame Number,Keypoints\n")

    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            _, frame = cap.read()
            if not _:
                break

            frame_count += 1
            frame, results = mediapipe_detection(frame, holistic)
            frame_height, frame_width = frame.shape[0], frame.shape[1]

            # Extract features
            status, face, pose, left_hand, right_hand = extract_keypoints(results)
            features = extract_features(face, pose, left_hand, right_hand, frame_width, frame_height) if status else features_old

            # Draw keypoints on the original frame
            for lm in features:
                x = int(lm[0])
                y = int(lm[1])
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            
            print(len(features))
            
            if np.sum(features[[26, 29, 32, 35], :]) and np.sum(features[[42, 45, 48, 51], :]) != 0.0:
                category = "non_blur"
                print(f"Frame {frame_count} is categorized as '{category}'")
            else:
                category = "blur"
                print(f"Frame {frame_count} is categorized as '{category}'")


            # # Classify images based on keypoints
            # has_left_keypoints = np.any(left_hand)
            # has_right_keypoints = np.any(right_hand)

            # if has_left_keypoints and has_right_keypoints:
            #     category = "both_keypoints"
            # elif has_left_keypoints:
            #     category = "left_keypoints"
            # elif has_right_keypoints:
            #     category = "right_keypoints"
            # else:
            #     category = "no_keypoints"

            # Save frame in the appropriate category folder
            category_folder = os.path.join(frame_output_folder, category)
            save_path_frame = os.path.join(category_folder, f"{video_file[:-4]}_frame_{frame_count}.jpg")
            cv2.imwrite(save_path_frame, frame)
            print(f"Frame {frame_count} saved in category '{category}' at: {save_path_frame}")

            # Save keypoints to the text file
            with open(keypoints_file, "a") as kp_file:
                kp_file.write(f"{frame_count},{features.tolist()}\n")

            # Normalize features for future processing
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            features = (features - mean) / std

            features_old = features  # Save for fallback

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
