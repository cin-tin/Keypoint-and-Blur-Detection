import numpy as np
import mediapipe as mp
import cv2
import os

# Initialize Mediapipe solutions
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Define paths
dataset_dir = "input video"
output_base = "ouput"

# Define main categories
main_categories = ["group1_all_detected", "group2_keypoints_absent"]

# Define subcategories for group 2
subcategories_group2 = ["absent_hands", "absent_left_hand", "absent_right_hand", "absent_pose", "absent_face"]

# Create output directories
for main_category in main_categories:
    os.makedirs(os.path.join(output_base, main_category), exist_ok=True)

for subcategory in subcategories_group2:
    os.makedirs(os.path.join(output_base, "group2_keypoints_absent", subcategory), exist_ok=True)

# Function to draw keypoints on the frame
def draw_keypoints(frame, keypoints, color=(255, 0, 0)):
    for kp in keypoints:
        if np.any(kp):  # Check if the keypoint is valid
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 4, color, -1)

# Function for Mediapipe detection
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results

# Detection functions
def extract_hands(results, frame_width, frame_height):
    right_hand = np.zeros((21, 2))
    left_hand = np.zeros((21, 2))

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            hand_landmarks = results.multi_hand_landmarks[idx]
            keypoints = np.array([[lm.x * frame_width, lm.y * frame_height] for lm in hand_landmarks.landmark])

            # Swap the perspectives
            if hand_label == 'Left':  # Originally left
                right_hand = keypoints  # Now considered right
            elif hand_label == 'Right':  # Originally right
                left_hand = keypoints  # Now considered left

    has_left = np.any(left_hand)
    has_right = np.any(right_hand)

    return has_left, has_right, left_hand, right_hand

def extract_pose_face(results, frame_width, frame_height):
    # Pose keypoints (selected indices)
    pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    pose_keypoints = np.zeros((8, 2))
    if results.pose_landmarks:
        pose_keypoints = np.array([[results.pose_landmarks.landmark[idx].x * frame_width,
                                    results.pose_landmarks.landmark[idx].y * frame_height] for idx in pose_indices])

    # Face keypoints (selected indices)
    face_indices = [4, 13, 17, 61, 68, 93, 151, 175, 207, 291, 298, 427, 454, 33, 133, 362, 263]
    face_keypoints = np.zeros((len(face_indices), 2))
    if results.face_landmarks:
        face_keypoints = np.array([[results.face_landmarks.landmark[idx].x * frame_width,
                                    results.face_landmarks.landmark[idx].y * frame_height] for idx in face_indices])

    has_pose = np.any(pose_keypoints)
    has_face = np.any(face_keypoints)

    return has_pose, pose_keypoints, has_face, face_keypoints

# Process videos
for video_file in os.listdir(dataset_dir):
    video_extensions = ['.mp4', '.mov', '.MOV', '.MP4']
    if not any(video_file.endswith(ext) for ext in video_extensions):
        continue

    video_path = os.path.join(dataset_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"Processing video: {video_file}")

    with mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2) as hands, \
         mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Skipping unreadable frame {frame_count} from {video_file}.")
                frame_count += 1
                continue

            frame_count += 1
            frame_height, frame_width = frame.shape[:2]

            try:
                # HANDS
                _, results_hands = mediapipe_detection(frame, hands)
                has_left, has_right, left_hand, right_hand = extract_hands(results_hands, frame_width, frame_height)
                draw_keypoints(frame, left_hand, color=(0, 255, 0))  # Green for swapped left hand
                draw_keypoints(frame, right_hand, color=(0, 0, 255))  # Blue for swapped right hand

                # POSE AND FACE
                _, results_holistic = mediapipe_detection(frame, holistic)
                has_pose, pose_keypoints, has_face, face_keypoints = extract_pose_face(results_holistic, frame_width, frame_height)
                draw_keypoints(frame, pose_keypoints, color=(255, 255, 0))  # Yellow for pose
                draw_keypoints(frame, face_keypoints, color=(255, 0, 255))  # Magenta for face

                # Categorize frames
                if has_left and has_right and has_pose and has_face:
                    category = "group1_all_detected"
                    save_path = os.path.join(output_base, category, f"{video_file[:-4]}_frame_{frame_count}.jpg")
                else:
                    category = "group2_keypoints_absent"

                    # Check for absent hands
                    if not has_left and not has_right:
                        subcategory = "absent_hands"
                    elif not has_left:
                        subcategory = "absent_left_hand"
                    elif not has_right:
                        subcategory = "absent_right_hand"
                    elif not has_pose:
                        subcategory = "absent_pose"
                    elif not has_face:
                        subcategory = "absent_face"
                    else:
                        subcategory = None

                    save_path = os.path.join(output_base, category, subcategory, f"{video_file[:-4]}_frame_{frame_count}.jpg")

                # Save frame
                if subcategory is not None or category == "group1_all_detected":
                    cv2.imwrite(save_path, frame)

            except Exception as e:
                print(f"Error processing frame {frame_count} of {video_file}: {e}")

    cap.release()
    cv2.destroyAllWindows()
