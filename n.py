import cv2
import numpy as np
import mediapipe as mp
import time
import handModule as hm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()


detector = hm.handDetector()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()

min_vol, max_vol = vol_range[0], vol_range[1]
vol_percent = 0
str_vol = ""

prev_time = time.time()
flag = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        continue

    frame = detector.mark_handlms(frame)
    id_coords_list = detector.coordinates(frame)
    cv2.rectangle(frame, (30, 100), (55, 300), (0, 0, 255), 2)

    if len(id_coords_list) != 0:
        # Extract landmarks
        thumb_x, thumb_y = id_coords_list[4][1], id_coords_list[4][2]
        middle_x, middle_y = id_coords_list[12][1], id_coords_list[12][2]
        ring_x, ring_y = id_coords_list[16][1], id_coords_list[16][2]
        pinky_x, pinky_y = id_coords_list[20][1], id_coords_list[20][2]
        wrist_x, wrist_y = id_coords_list[0][1], id_coords_list[0][2]

        # Draw circles on key points
        cv2.circle(frame, (thumb_x, thumb_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (middle_x, middle_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (ring_x, ring_y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (pinky_x, pinky_y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (wrist_x, wrist_y), 6, (0, 255, 0), -1)

        # Calculate distances
        dist_wrist_ring = int(math.hypot(wrist_x - ring_x, wrist_y - ring_y))
        dist_pinky_ring = int(math.hypot(pinky_x - ring_x, pinky_y - ring_y))
        
        # Gesture conditions
        if dist_pinky_ring > 60:
            flag = 1
        if 40 <= dist_wrist_ring <= 85:
            flag = 0

        dist = int(np.hypot(thumb_x - middle_x, thumb_y - middle_y))

        # Draw lines
        cv2.line(frame, (wrist_x, wrist_y), (ring_x, ring_y), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (pinky_x, pinky_y), (ring_x, ring_y), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, (thumb_x, thumb_y), (middle_x, middle_y), (0, 0, 255), 2, cv2.LINE_AA)

        # Volume Control
        min_dist, max_dist = 0, 150
        if flag != 1:
            interpreted_vol = np.interp(dist, (min_dist, max_dist), (min_vol, max_vol))
            volume.SetMasterVolumeLevel(interpreted_vol, None)
            vol_percent = np.interp(interpreted_vol, (min_vol, max_vol), (295, 103))
            vol_bar = np.interp(interpreted_vol, (min_vol, max_vol), (0, 100))
            str_vol = f"{int(vol_bar)} %"
            cv2.rectangle(frame, (32, int(vol_percent)), (53, 298), (0, 250, 0), -1)

    # Show Volume Bar
    if flag == 1:
        cv2.rectangle(frame, (32, int(vol_percent)), (53, 298), (0, 250, 0), -1)

    # FPS Calculation
    curr_time = time.time()
    if curr_time != prev_time:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0
    prev_time = curr_time

    # Display FPS & Volume
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str_vol, (20, 340), font, 1, (255, 0, 244), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Volume Control", frame)

    # Exit on Esc key or window close
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Gesture Volume Control", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
