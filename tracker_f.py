import cv2
import numpy as np
import sys
import math

# Define Tag
id_to_find = 72
marker_size = 10  # [cm]

# Function to check if a matrix is a valid rotation matrix
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Function to calculate rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# Get the camera calibration path
calib_path = ""
try:
    camera_matrix = np.loadtxt(calib_path + 'camera_matrix.txt', delimiter=',')
    camera_distortion = np.loadtxt(calib_path + 'Distortion_coeff.txt', delimiter=',')
except Exception as e:
    print(f"Error loading calibration files: {e}")
    sys.exit()

# Initialize video capture with V4L2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    sys.exit()

# Set camera size as calibrated
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Generator function to process frames with Aruco markers
def process_frames():
    while True:
        # Read the camera frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find all the ArUco markers in the image
        try:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        except Exception as e:
            print(f"Error detecting markers: {e}")
            continue

        if ids is not None and id_to_find in ids:
            # Get the index of the marker to find
            try:
                index = np.where(ids == id_to_find)[0][0]
            except Exception as e:
                print(f"Error indexing ids: {e}")
                continue

            # Estimate pose of the marker
            try:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[index], marker_size, camera_matrix, camera_distortion)
            except Exception as e:
                print(f"Error estimating pose: {e}")
                continue

            # Ensure rvec and tvec are in the correct shape
            rvec = np.squeeze(rvec)
            tvec = np.squeeze(tvec)

            # Draw detected marker and put a reference frame over it
            try:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)
            except Exception as e:
                print(f"Error drawing markers/axis: {e}")

            # Print marker position in camera frame
            str_position = f"MARKER Position x={int(tvec[0])}  y={int(tvec[1])}  z={int(tvec[2])}"
            cv2.putText(frame, str_position, (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Obtain rotation matrix tag->camera
            try:
                R_ct, _ = cv2.Rodrigues(rvec)
                R_tc = R_ct.T
            except Exception as e:
                print(f"Error computing rotation matrix: {e}")
                continue

            # Get attitude in terms of euler 321 (Needs to be flipped first)
            try:
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_tc)
            except Exception as e:
                print(f"Error computing euler angles: {e}")
                continue

            # Print marker's attitude respect to camera frame
            str_attitude = f"MARKER Attitude r={int(math.degrees(roll_marker))}  p={int(math.degrees(pitch_marker))}  y={int(math.degrees(yaw_marker))}"
            cv2.putText(frame, str_attitude, (0, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Get Position and attitude of the camera respect to the marker
            try:
                pos_camera = -R_tc @ np.matrix(tvec).T
            except Exception as e:
                print(f"Error computing camera position: {e}")
                continue

            str_position = f"CAMERA Position x={int(pos_camera[0])}  y={int(pos_camera[1])}  z={int(pos_camera[2])}"
            cv2.putText(frame, str_position, (0, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Get attitude of the camera respect to the frame
            try:
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_tc)
            except Exception as e:
                print(f"Error computing camera euler angles: {e}")
                continue

            str_attitude = f"CAMERA Attitude r={int(math.degrees(roll_camera))}  p={int(math.degrees(pitch_camera))}  y={int(math.degrees(yaw_camera))}"
            cv2.putText(frame, str_attitude, (0, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Yield the processed frame
        yield frame

# Main loop to display frames
for frame in process_frames():
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
