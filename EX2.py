import cv2
import numpy as np
import pandas as pd
import time

# Example camera matrix and distortion coefficients
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
df = pd.DataFrame({"id": [], "pos": [], "dist": [], "angle": [], "pitch-roll": []})


def detect_datamatrix(frame):
    df_temp = pd.DataFrame(
        {"id": [], "pos": [], "dist": [], "angle": [], "pitch-roll": []}
    )

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximate contour with accuracy proportional to the contour perimeter
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Skip small or non-quadrilateral contours
        if len(approx) != 4 or cv2.contourArea(approx) < 1000:
            continue

        # Draw the contours on the frame

        # Extract the region of interest
        x, y, w, h = cv2.boundingRect(approx)
        roi = gray[y : y + h, x : x + w]

        # Check if the ROI has a Data Matrix-like structure
        if is_datamatrix_like(roi):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Load the dictionary that was used to generate the markers
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

            # Initialize the detector parameters using default values
            parameters = cv2.aruco.DetectorParameters()

            # Detect the markers in the frame
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray_frame, aruco_dict, parameters=parameters
            )

            # Define the marker side length (in meters)
            marker_length = 0.05  # Example length, change it according to your markers

            # Perform pose estimation
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            # Draw the detected markers and their poses
            if ids is not None:
                for i in range(len(ids)):
                    cv2.polylines(frame, [approx], True, (0, 255, 0), 3)
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    # Calculate distance to the camera
                    distance = np.linalg.norm(tvecs[i])
                    rvec_matrix, _ = cv2.Rodrigues(rvecs[i])
                    sy = np.sqrt(rvec_matrix[0, 0] ** 2 + rvec_matrix[1, 0] ** 2)
                    singular = sy < 1e-6
                    if not singular:
                        yaw = np.arctan2(rvec_matrix[2, 0], rvec_matrix[2, 2])
                        pitch = np.arctan2(-rvec_matrix[2, 1], sy)
                        roll = np.arctan2(rvec_matrix[1, 0], rvec_matrix[0, 0])
                    else:
                        yaw = np.arctan2(-rvec_matrix[1, 2], rvec_matrix[1, 1])
                        pitch = np.arctan2(-rvec_matrix[2, 1], sy)
                        roll = 0

                    # Calculate the center of the frame
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2

                    # Calculate the center of the detected Data Matrix code
                    datamatrix_center_x = x + w // 2
                    datamatrix_center_y = y + h // 2

                    # Determine instructions based on the position of the Data Matrix code
                    instructions = "Id: " + str(ids[i]) + "\n"
                    # Determine rotation instructions based on the roll angle
                    if yaw > 0:
                        instructions += "1. Rotate Right "
                        instructions += "\n"
                        print(yaw)
                    elif yaw < 0:
                        instructions += "2. Rotate Left "
                        instructions += "\n"
                        print(yaw)

                    if datamatrix_center_x < frame_center_x - 100:
                        instructions += "3. Move Left "
                        instructions += "\n"
                    elif datamatrix_center_x > frame_center_x + 100:
                        instructions += "4. Move Right "
                        instructions += "\n"
                    if datamatrix_center_y < frame_center_y - 100:
                        instructions += "5. Move Up "
                        instructions += "\n"
                    elif datamatrix_center_y > frame_center_y + 100:
                        instructions += "6. Move Down "
                        instructions += "\n"

                    df_temp.loc[-1] = [
                        ids[i],
                        (x, y, w, h),
                        distance,
                        (np.degrees(yaw)),
                        (pitch, roll),
                    ]
                    print(instructions)
                    cv2.putText(
                        frame,
                        instructions,
                        (0, 100),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 0, 0),
                        2,
                    )
                    df_temp.index = df_temp.index + 1
                    df_temp = df_temp.sort_index()
    return frame, df_temp


def is_datamatrix_like(roi):
    # Simple check for Data Matrix structure (e.g., based on size and aspect ratio)
    h, w = roi.shape
    if 0.8 < float(w) / h < 1.2:
        return True
    return False


# Open the video file
input_video_path = "challengeB.mp4"
output_video_path = "output_video_with_datamatrix.mp4"
cap = cv2.VideoCapture(0)
# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    #time.sleep(0.5)
    if not ret:
        break

    # Detect and mark Data Matrix codes in the current frame
    marked_frame, res_df = detect_datamatrix(frame)
    # print(res_df)
    df = df.append(res_df)
    # Write the frame to the output video
    cv2.imshow("frame", marked_frame)
    out.write(marked_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
df.to_csv("out.csv")
# Release the video capture and writer objects
cap.release()
out.release()
