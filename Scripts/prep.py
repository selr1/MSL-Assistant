import cv2
import mediapipe as mp
import pandas as pd
import os

# Paths for dataset and output
DATASET_ROOT = '../Datasets'
OUTPUT_CSV = '../CSVs/preprocessed.csv'

# Setup MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)
# Iman was here
def main():
    all_extracted_data = []
    
    # Walk through every folder to find images
    for root, dirs, files in os.walk(DATASET_ROOT):
        
        # Folder name is label (e.g., 'MAKAN')
        folder_label = os.path.basename(root).upper()
        
        # Only look for image files
        valid_images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not valid_images:
            continue

        for img_name in valid_images:
            img_path = os.path.join(root, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Extract 21 landmarks if hand is found
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_row = []
                    
                    # Store x, y, z for all 21 points (63 features total)
                    for lm in hand_landmarks.landmark:
                        landmark_row.extend([lm.x, lm.y, lm.z])
                    
                    # Add label and save to list
                    landmark_row.append(folder_label)
                    all_extracted_data.append(landmark_row)
                    
                    # One hand per image is enough
                    break 

    headers = [f'pt{i}_{ax}' for i in range(21) for ax in ['x','y','z']] + ['label']
    df = pd.DataFrame(all_extracted_data, columns=headers)

    # Create folder if missing
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Keeping final summary results
    print("\n" + "RESULTS")
    print(f"Samples: {len(df)}")
    print(f"Unique Signs: {df['label'].nunique()}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()