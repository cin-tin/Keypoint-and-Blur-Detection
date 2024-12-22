import os
import json
import cv2
import mediapipe as mp
import numpy as np
import gc

# Initialize Mediapipe models
mp_hands = mp.solutions.hands

def process_landmarks(landmarks):
    """Extract x and y coordinates from Mediapipe landmarks."""
    x_list, y_list = [], []
    if landmarks:
        for landmark in landmarks.landmark:
            x_list.append(landmark.x)
            y_list.append(landmark.y)
    return x_list, y_list

def draw_keypoints_on_image(image, landmarks, frame_width, frame_height):
    """Draw detected hand keypoints on the image."""
    if landmarks:
        for lm in landmarks.landmark:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Draw keypoints
    return image

def save_hand_keypoints_to_text(uid, hand1_x, hand1_y, hand2_x, hand2_y, frame_names, save_dir):
    """Save hand keypoints in a human-readable text file with frame names and keypoint numbers."""
    output_file = os.path.join(save_dir, f"{uid}_hand_keypoints.txt")
    
    with open(output_file, "w") as f:
        f.write(f"UID: {uid}\n\n")
        
        for frame_idx, frame_name in enumerate(frame_names):
            f.write(f"Frame: {frame_name}\n")
            
            # Hand 1 keypoints
            f.write("Hand 1 Keypoints:\n")
            present_hand1_keypoints = [i + 1 for i, (hx, hy) in enumerate(zip(hand1_x[frame_idx], hand1_y[frame_idx])) if not (np.isnan(hx) or np.isnan(hy))]
            f.write(f"    Present Keypoints: {present_hand1_keypoints}\n")
            
            # Hand 2 keypoints
            f.write("Hand 2 Keypoints:\n")
            present_hand2_keypoints = [i + 1 for i, (hx, hy) in enumerate(zip(hand2_x[frame_idx], hand2_y[frame_idx])) if not (np.isnan(hx) or np.isnan(hy))]
            f.write(f"    Present Keypoints: {present_hand2_keypoints}\n")
            
            f.write("\n")
        
        f.write(f"Number of Frames Processed: {len(frame_names)}\n")

    print(f"Hand keypoints saved in text format at: {output_file}")

def extract_keypoints_from_images(image_folder, save_dir, frames_with_keypoints_dir):
    """Process a folder of images to extract hand keypoints, saving to JSON, text files, and overlay images."""
    # Mediapipe instances
    hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)
    
    # Initialize storage for keypoints
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []
    frame_names = []

    # Process each image in the folder
    os.makedirs(frames_with_keypoints_dir, exist_ok=True)
    for image_name in sorted(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        if not os.path.isfile(image_path) or not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = image.shape[:2]

        # Process hands
        hand_results = hands.process(image_rgb)

        # Initialize hand keypoints
        hand1_x, hand1_y, hand2_x, hand2_y = [], [], [], []

        # Process hands if detected
        if hand_results.multi_hand_landmarks:
            if len(hand_results.multi_hand_landmarks) > 0:
                hand1_x, hand1_y = process_landmarks(hand_results.multi_hand_landmarks[0])
                image = draw_keypoints_on_image(image, hand_results.multi_hand_landmarks[0], frame_width, frame_height)
            if len(hand_results.multi_hand_landmarks) > 1:
                hand2_x, hand2_y = process_landmarks(hand_results.multi_hand_landmarks[1])
                image = draw_keypoints_on_image(image, hand_results.multi_hand_landmarks[1], frame_width, frame_height)

        # Handle missing data by filling with NaNs
        hand1_x = hand1_x if hand1_x else [np.nan] * 21
        hand1_y = hand1_y if hand1_y else [np.nan] * 21
        hand2_x = hand2_x if hand2_x else [np.nan] * 21
        hand2_y = hand2_y if hand2_y else [np.nan] * 21

        # Append keypoints to lists
        hand1_points_x.append(hand1_x)
        hand1_points_y.append(hand1_y)
        hand2_points_x.append(hand2_x)
        hand2_points_y.append(hand2_y)
        frame_names.append(image_name)

        # Save frame with detected keypoints
        save_frame_path = os.path.join(frames_with_keypoints_dir, image_name)
        cv2.imwrite(save_frame_path, image)
        print(f"Frame with keypoints saved at: {save_frame_path}")

    # Prepare data for saving
    uid = os.path.basename(image_folder)
    save_data = {
        "uid": uid,
        "hand1_x": hand1_points_x,
        "hand1_y": hand1_points_y,
        "hand2_x": hand2_x,
        "hand2_y": hand2_y,
        "frame_names": frame_names,
    }

    # Save data to JSON
    os.makedirs(save_dir, exist_ok=True)
    json_file = os.path.join(save_dir, f"{uid}_hand_keypoints.json")
    with open(json_file, "w") as f:
        json.dump(save_data, f)
    print(f"Hand keypoints saved in JSON format at: {json_file}")

    # Save data to text file
    save_hand_keypoints_to_text(
        uid,
        hand1_points_x,
        hand1_points_y,
        hand2_points_x,
        hand2_points_y,
        frame_names,
        save_dir,
    )

    # Clean up
    hands.close()
    del hands, save_data
    gc.collect()

# Example usage
image_folder = "/home/tenet/Desktop/ISL-NEW/crop/blur_test/test_images"  # Replace with the path to your folder of images
output_dir = "/home/tenet/Desktop/ISL-NEW/crop/blur_test/keypoints"       # Replace with your desired output directory
frames_with_keypoints_dir = "/home/tenet/Desktop/ISL-NEW/crop/blur_test/keypoints"  # Directory for frames with keypoints
extract_keypoints_from_images(image_folder, output_dir, frames_with_keypoints_dir)
