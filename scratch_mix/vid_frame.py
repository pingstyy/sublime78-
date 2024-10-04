import cv2
import os
import numpy as np

# Open the video file
video_path = 'path/to/video.mp4'
cap = cv2.VideoCapture(video_path)

# Create a folder to store the frames
frame_folder = 'frames'
if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

# Initialize the frame counter
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a NumPy array
    frame_array = np.array(frame)

    # Save the frame as a PNG image
    frame_path = os.path.join(frame_folder, f'frame_{frame_count:06d}.png')
    cv2.imwrite(frame_path, frame)

    # Increment the frame counter
    frame_count += 1

# Release the video capture object
cap.release()