import cv2
import cv2.aruco as aruco

# Define the Aruco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Open the webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters)

    # If markers are detected, print their IDs and draw bounding boxes
    if ids is not None:
        for detected_id, corner in zip(ids.flatten(), corners):
            print(f"Detected Aruco ID: {detected_id}")
            # Draw the bounding box around the marker
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

            # Optionally, draw a rectangle around the detected marker
            # Get the coordinates of the corners
            corner_points = corner[0]  # Each corner is a 4-point array

            # Draw the bounding box using polylines
            cv2.polylines(frame, [corner_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
