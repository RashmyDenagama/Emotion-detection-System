import cv2
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import deque

# Load pre-trained model for emotion detection
model = load_model('emotion_detection_model.h5')
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Initialize webcam for real-time emotion detection
cap = cv2.VideoCapture(0)

# Create a list to store emotion data (emotion and timestamp)
emotion_data = []

# Create the 'output' directory to save files if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Load Haar Cascade for face detection (pre-trained classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up video writer to save the processed video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('output/emotion_video.avi', fourcc, 20.0, (640, 480))

# Initialize deque for smoothing predictions over recent frames (smoothing window of 10 frames)
recent_predictions = deque(maxlen=10)

# Start the webcam feed loop for real-time emotion detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # If frame capture fails, break the loop
    
    # Convert the frame to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract face region and preprocess for model input
        face = frame[y:y + h, x:x + w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))  # Resize face to 48x48 pixels
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0) / 255.0  # Normalize the image

        # Get emotion prediction from the model
        emotion_probabilities = model.predict(face_array)
        max_index = np.argmax(emotion_probabilities[0])  # Get the index of the highest probability
        predicted_emotion = emotion_labels[max_index]  # Get the emotion label corresponding to the highest probability

        # Add prediction to the list of recent predictions for smoothing
        recent_predictions.append(predicted_emotion)
        smoothed_emotion = max(set(recent_predictions), key=recent_predictions.count)  # Most common emotion in the window

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, smoothed_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Append the emotion and timestamp to the emotion data list
        emotion_data.append({'Emotion': smoothed_emotion, 'Time': pd.Timestamp.now()})

    # Write the frame with annotations to the video file
    video_out.write(frame)

    # Display the annotated frame in a window
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the emotion data to a CSV file after the loop ends
df = pd.DataFrame(emotion_data)
df.to_csv('output/detected_emotions.csv', index=False)

# Rating logic: map each emotion to a rating value
emotion_to_rating = {
    'Happiness': 5,
    'Surprise': 5,
    'Anger': 1,
    'Disgust': 1,
    'Fear': 1,
    'Sadness': 1,
    'Neutral': 3
}

# Load the detected emotions CSV file
emotion_df = pd.read_csv('output/detected_emotions.csv')

# Add a 'Rating' column based on the emotion-to-rating mapping
emotion_df['Rating'] = emotion_df['Emotion'].map(emotion_to_rating)

# Calculate the average rating for the entire video based on the detected emotions
average_rating = emotion_df['Rating'].mean()

# Print the average rating for the video
print(f"Average Rating for the Emotion Detection: {average_rating:.2f}")

# Ensure the 'output' directory exists before saving the ratings data
if not os.path.exists('output'):
    os.makedirs('output')

# Save the ratings data to a new CSV file
emotion_df.to_csv('output/emotion_with_ratings.csv', index=False)

# Release the webcam and video writer resources, and close all OpenCV windows
cap.release()
video_out.release()
cv2.destroyAllWindows()

# Print final messages about saved files
print("Emotion data saved to: output/detected_emotions.csv")
print("Video saved to: output/emotion_video.avi")
print("Emotion ratings saved to: output/emotion_with_ratings.csv")
 