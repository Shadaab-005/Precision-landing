import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard
chessboard_size = (9, 6)
# Define the size of a square in your defined unit (e.g., millimeters)
square_size = 25  # Adjust as necessary

# Prepare object points based on the actual size of the chessboard squares
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Load calibration images
images = glob.glob('calibration images/*.jpg')  # Adjust the path as necessary

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to get the camera matrix, distortion coefficients, etc.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)

# Save the calibration result for later use
np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

# Optionally, you can undistort an image using the calibration data
def undistort_image(img_path):
    # Load the saved calibration data
    with np.load('camera_calibration.npz') as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image to the valid area
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    return undistorted_img

# Example usage
undistorted_image = undistort_image('test_image.jpg')
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
