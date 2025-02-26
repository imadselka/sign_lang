import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class SignLanguageApp:
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
        
        # Labels for ASL alphabet (Sign Language MNIST classes)
        # MNIST dataset uses 0-25 for A-Z (no J or Z as they require motion)
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y', 'Z']
        
        # Model path
        self.model_path = 'sign_language_model.h5'
        self.model = None
    
    def load_mnist_data(self, train_csv_path, test_csv_path=None):
        """Load data from Sign Language MNIST CSV files"""
        print(f"Loading training data from {train_csv_path}...")
        
        # Load training data
        train_df = pd.read_csv(train_csv_path)
        
        # Separate labels and pixel values
        y_train = train_df['label'].values
        X_train = train_df.drop('label', axis=1).values
        
        # Reshape images to 28x28 (original MNIST format)
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # Test data (if provided)
        X_test = None
        y_test = None
        if test_csv_path and os.path.exists(test_csv_path):
            print(f"Loading test data from {test_csv_path}...")
            test_df = pd.read_csv(test_csv_path)
            y_test = test_df['label'].values
            X_test = test_df.drop('label', axis=1).values
            X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        else:
            # Split training data to create validation set
            print("No test file provided, splitting training data for validation...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train, y_train, X_test, y_test
    
    def create_model(self):
        """Create a CNN model for sign language recognition"""
        model = tf.keras.Sequential([
            # First convolution layer and pooling
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Second convolution layer and pooling
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Third convolution layer and pooling
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, train_csv_path, test_csv_path=None, epochs=10):
        """Train the model using the MNIST dataset"""
        # Load data
        X_train, y_train, X_test, y_test = self.load_mnist_data(train_csv_path, test_csv_path)
        
        # Create model
        self.model = self.create_model()
        print(self.model.summary())
        
        # Train model
        print(f"Training model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=64,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return history
    
    def load_model(self, model_path=None):
        """Load a previously trained model"""
        if model_path:
            self.model_path = model_path
        
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found at {self.model_path}")
            return False
    
    def _preprocess_frame(self, frame, hand_landmarks):
        """Extract and preprocess the hand region from a frame based on landmarks"""
        h, w, c = frame.shape
        
        # Extract landmark coordinates
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
        
        # Get bounding box
        x_min, y_min = np.min(landmarks, axis=0).astype(int)
        x_max, y_max = np.max(landmarks, axis=0).astype(int)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract hand image
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None, (x_min, y_min, x_max, y_max)
        
        # Convert to grayscale (MNIST dataset is grayscale)
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28 (MNIST format)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize and reshape for model input
        normalized = resized.astype('float32') / 255.0
        preprocessed = normalized.reshape(1, 28, 28, 1)
        
        return preprocessed, (x_min, y_min, x_max, y_max)
    
    def run_recognition(self):
        """Run real-time sign language recognition"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam")
            return
        
        print("Starting sign language recognition...")
        print("Press 'q' to quit, 'a' to add sign to text")
        
        # Text area for displaying recognized signs
        text_sequence = ""
        last_prediction = None
        prediction_counter = 0
        stable_threshold = 5  # Number of consistent predictions needed
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Display text sequence at the top
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, text_sequence[-40:], (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_prediction = None
            
            # Process hands if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Preprocess hand region
                    preprocessed, (x_min, y_min, x_max, y_max) = self._preprocess_frame(frame, hand_landmarks)
                    
                    if preprocessed is not None:
                        # Make prediction
                        try:
                            prediction = self.model.predict(preprocessed, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class]
                            
                            # Get label
                            if predicted_class < len(self.labels):
                                predicted_sign = self.labels[predicted_class]
                                label = f"{predicted_sign} ({confidence:.2f})"
                                current_prediction = predicted_sign
                                
                                # Display prediction
                                cv2.putText(frame, label, (x_min, y_min - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            else:
                                print(f"Predicted class {predicted_class} out of range")
                        except Exception as e:
                            print(f"Prediction error: {e}")
            
            # Update stable prediction
            if current_prediction == last_prediction and current_prediction is not None:
                prediction_counter += 1
            else:
                prediction_counter = 0
                last_prediction = current_prediction
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a') and last_prediction is not None and prediction_counter >= stable_threshold:
                text_sequence += last_prediction
                prediction_counter = 0  # Reset after adding
            
            # Display instructions
            cv2.putText(frame, "Show a sign with your hand", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'a' to add sign to text, 'q' to quit", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Sign Language Recognition", frame)
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    print("Sign Language MNIST Recognition App")
    print("----------------------------------")
    
    app = SignLanguageApp()
    
    # Hardcoded paths to your dataset
    train_path = "C:/Users/imads/Downloads/signLang_dataset/sign_mnist_train.csv"
    test_path = "C:/Users/imads/Downloads/signLang_dataset/sign_mnist_test.csv"
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Dataset files not found!")
        return
        
    # Train model with the specified datasets
    print("Training model with the provided datasets...")
    app.train_model(train_path, test_path, epochs=10)
    
    while True:
        print("\nMenu:")
        print("1. Run sign language recognition")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == "1":
            app.run_recognition()
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()