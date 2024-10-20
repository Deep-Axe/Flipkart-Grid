from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import cv2.aruco as aruco

app = Flask(__name__)

# Aruco marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

@app.route('/detect_aruco', methods=['POST'])
def detect_aruco():
    data = request.json
    image_data = data['image']
    
    # Decode the base64 image
    image_data = image_data.split(',')[1]
    image = base64.b64decode(image_data)
    np_image = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    if frame is None:
        print('Failed to decode image')
        return jsonify({'aruco_id': None})
    
    # Log the shape of the received frame
    print(f'Received frame shape: {frame.shape}')

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Aruco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Log detected corners and IDs
    print(f'Detected corners: {corners}, Detected IDs: {ids}')
    
    if ids is not None:
        detected_id = int(ids[0][0])
    else:
        detected_id = None
    
    return jsonify({'aruco_id': detected_id})


if __name__ == '__main__':
    app.run(debug=True)