import cv2 
import mediapipe as mp
import pyautogui
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict

# Function to control system volume
def set_system_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(volume_level, None)

# Track previous positions of the hands
prev_x = 0
prev_y = 0  # Track previous y-coordinate

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands model
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize play/pause flag
play_pause = False

def count_fingers(lst):
    cnt = 0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        cnt += 1

    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        cnt += 1

    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        cnt += 1

    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        cnt += 1

    if (lst.landmark[5].x*100 - lst.landmark[4].x*100) > 6:
        cnt += 1

    return cnt

while True:
    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hand_obj.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:
        for hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Return whether it is Right or Left Hand
            label = MessageToDict(results.multi_handedness[hand])['classification'][0]['label']

            if label == 'Left':
                # Count fingers
                cnt = count_fingers(hand_landmarks)

                # Adjust volume based on finger position
                if cnt == 5:
                    # Play/pause if five fingers are shown
                    if not play_pause:
                        pyautogui.press('playpause')
                        play_pause = True
                else:
                    play_pause = False

                # Extracting the y-coordinate of index finger tip
                y_8 = int(hand_landmarks.landmark[8].y * img.shape[0])
                
                # Adjust volume based on vertical movement of the index finger
                volume_range = (1, 0)  # Adjust volume from 0 to 1 (inverted)
                y_min, y_max = 50, 300  # Adjust these values based on your camera resolution

                # Map finger position to volume level (invert the range)
                volume_level = np.interp(y_8, [y_min, y_max], volume_range)

                # Set the volume level
                set_system_volume(volume_level)

            elif label == 'Right':
                # Extracting the x-coordinate of index finger tip
                x_8 = int(hand_landmarks.landmark[8].x * img.shape[1])
                y_8 = int(hand_landmarks.landmark[8].y * img.shape[0])  # Extract y-coordinate of index finger tip

                # If previous x-coordinate is 0, initialize it
                if prev_x == 0:
                    prev_x = x_8

                # If previous y-coordinate is 0, initialize it
                if prev_y == 0:
                    prev_y = y_8

                # Check if hand is sliding from right to left
                if x_8 < prev_x:
                    # Play next track on Spotify
                    pyautogui.press('nexttrack')

                    # Introduce a delay to allow the window to become responsive again
                    time.sleep(0.5)  # Adjust the duration as needed
                elif x_8 > prev_x:
                    # Play previous track on Spotify
                    pyautogui.press('prevtrack')

                    time.sleep(0.5)

                # Check if hand is sliding from top to bottom
                if y_8 > prev_y:
                    # Scroll down
                    pyautogui.scroll(-200)  # Adjust the scroll amount as needed
                elif y_8 < prev_y:
                    # Scroll up
                    pyautogui.scroll(200)  # Adjust the scroll amount as needed

                # Update previous coordinates
                prev_x = x_8
                prev_y = y_8

    # Draw landmarks on the image
    if results.multi_hand_landmarks:
        drawing.draw_landmarks(img, hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the image
    cv2.imshow("Hand Gesture Recognition", img)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
