from ultralytics import YOLO
import cv2

# Path to the best model
best_model_path = 'D:/study/OLD/acv/Atm_system/runs/train/train/weights/best.pt'

# Load the best model
model = YOLO(best_model_path)

print("successful")

# Initialize video capture (0 for the default webcam, or provide a video file path)
video_capture = cv2.VideoCapture('D:/study/OLD/acv/Atm_system/Videos/test_video.mp4')  # Use 0 for webcam or replace with video file path
video_capture = cv2.VideoCapture(0)

output_path = 'D:/study/OLD/acv/Atm_system/runs/test_output/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fps = 1
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # frame = cv2.resize(frame, (640,480))
    
    if not ret:
        break  # Break the loop if there are no frames to read

    # Perform inference on the frame
    results = model.predict(source=frame)

    # Extract the annotated frame (you can also process results further if needed)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-Time Inference', annotated_frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()


print("done")