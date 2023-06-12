import cv2
import os
import time

output_directory = r"E:\Projects\liveness_integration\train_img\spoof\Preetha"
frame_count = 125

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Count the number of existing frames in the directory
existing_frames = len(os.listdir(output_directory))

video = cv2.VideoCapture(0)  # Use index 0 for the default webcam

count = existing_frames + 1  # Start counting from the next frame after existing frames

# Display the countdown on the video capture display for 3 seconds
start_time = time.time()
while time.time() - start_time < 3:
    countdown = int(3 - (time.time() - start_time))
    print(f"Countdown: {countdown}")
    ret, frame = video.read()
    if ret:
        countdown_text = f"Countdown: {countdown}"
        cv2.putText(frame, countdown_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)

# Extract frames after the countdown finishes
while count <= existing_frames + frame_count:
    success, frame = video.read()
    name = os.path.join(output_directory, f"frame_{count}.jpg")

    if success:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        cv2.imwrite(name, frame)
        print(f"Frame {count} Extracted Successfully")
        count += 1
    else:
        break

video.release()
cv2.destroyAllWindows()
