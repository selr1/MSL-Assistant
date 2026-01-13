import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np

# Load models and scaler
rf_model = joblib.load('./rf_model.pkl')
svm_model = joblib.load('./svm_model1.5.pkl')
scaler = joblib.load('./scaler.pkl')

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)
# iman was here
def main():
    cap = cv2.VideoCapture(0)
    current_model = rf_model
    model_name = "Random Forest"

    print(f"Active Model: {model_name}")
    print("Press 's' to switch models, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract 63 coordinates
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                
                # Prepare data
                hand_cols = [f'pt{i}_{ax}' for i in range(21) for ax in ['x','y','z']]
                df_input = pd.DataFrame([row], columns=hand_cols)
                
                # Scale and predict
                scaled_input = scaler.transform(df_input)
                probs = current_model.predict_proba(scaled_input)
                
                # Get prediction even if confidence is low
                confidence = np.max(probs)
                prediction = current_model.classes_[np.argmax(probs)]

                # Always display result
                display_text = f"{prediction} ({confidence*100:.1f}%)"
                
                # Green for high confidence, red for low
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                
                cv2.putText(frame, f"Model: {model_name}", (10, 30), 1, 1, (255, 255, 255), 1)
                cv2.putText(frame, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            # Show message when no hand is detected
            cv2.putText(frame, "No hand detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('MediSign', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if model_name == "Random Forest":
                current_model = svm_model
                model_name = "SVM"
            else:
                current_model = rf_model
                model_name = "Random Forest"
            print(f"Switched to: {model_name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
