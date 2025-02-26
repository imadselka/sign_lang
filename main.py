import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Labels for ASL alphabet (excluding J and Z which require motion)
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                      'V', 'W', 'X', 'Y']
        
        # Load the pre-trained model
        self.model = tf.keras.models.load_model('sign_language_model.h5')
        print("Model loaded successfully!")

    def detect_hands(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        return results

    def preprocess_hand(self, frame, hand_landmarks):
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Get hand coordinates
        coords = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        
        # Get bounding box
        x_min, y_min = np.min(coords, axis=0).astype(int)
        x_max, y_max = np.max(coords, axis=0).astype(int)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract and process hand image
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None, None
        
        # Convert to grayscale and resize
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize and reshape
        processed = resized.astype('float32') / 255.0
        processed = processed.reshape(1, 28, 28, 1)
        
        return processed, (x_min, y_min, x_max, y_max)

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = self.detect_hands(frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Process hand image
                    processed_hand, bbox = self.preprocess_hand(frame, hand_landmarks)
                    
                    if processed_hand is not None:
                        # Make prediction
                        prediction = self.model.predict(processed_hand, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        
                        if predicted_class < len(self.labels):
                            predicted_letter = self.labels[predicted_class]
                            label = f"{predicted_letter} ({confidence:.2f})"
                            
                            # Draw prediction and bounding box
                            x_min, y_min, x_max, y_max = bbox
                            cv2.putText(frame, label, (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                        (0, 255, 0), 2)
            
            # Show instructions
            cv2.putText(frame, "Show hand sign in frame (Press 'q' to quit)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Sign Language Detection', frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = SignLanguageDetector()
    detector.run_detection()

if __name__ == "__main__":
    main()