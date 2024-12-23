import cv2
import os

# Path to the cascade classifier
cascade_classifier_path = "C:\\Users\\user\\Desktop\\Edgematrix\\xml files\\haarcascade_fullbody.xml"
cascade_classifier = cv2.CascadeClassifier(cascade_classifier_path)

# Check if the cascade classifier loaded correctly
if cascade_classifier.empty():
    print("Error: Could not load cascade classifier. Please check the path.")
    exit()

# Function to detect objects
def detect_objects(frame, cascade_classifier, output_folder, frame_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Histogram equalization to improve contrast

    objects = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Smaller step size for scaling
        minNeighbors=7,    # Increase to reduce false positives
        minSize=(50, 50),  # Minimum size for detected objects
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in objects:
        # Draw rectangles around detected objects
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the frame with detected objects
        cv2.imwrite(os.path.join(output_folder, f'detected_full_body_{frame_count}.png'), frame)

    return frame

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window for display
cv2.namedWindow('Upper Body Detection')

# Create output directory for detected images
output_folder = 'detected_upper_bodies'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

# Main loop for upper body detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects and update the frame count
    frame = detect_objects(frame, cascade_classifier, output_folder, frame_count)
    frame_count += 1

    # Display the frame
    cv2.imshow('Upper Body Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
