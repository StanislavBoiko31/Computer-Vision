
'''
Приклади реалізації процесів object tracking з використанням методів OpenCV:

https://hackthedeveloper.com/object-tracking-opencv-python/
https://broutonlab.com/blog/opencv-object-tracking
https://machinelearningknowledge.ai/learn-object-tracking-in-opencv-python-with-code-examples/
https://docs.opencv.org/3.4/d9/df8/group__tracking.html

1. MeanShift : непараметричний алгоритм, який використовує кольорову гістограму (аналіз кольорового простору) для відстеження об’єктів.
2. CamShift : розширення MeanShift, який адаптується до змін розміру та орієнтації об’єкта.

Package            Version
------------------ -----------
numpy              1.24.1
opencv-python      3.4.18.65

'''

import cv2
import numpy as np

# 1. MeanShift
def MeanShift (cap):

    # Read the first frame
    ret, frame = cap.read()
    '''
    WARNING! click ENTER
    '''
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Initialize the tracker
    roi = frame[y:y + h, x:x + w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 256], 1)
        '''
        Apply the MeanShift algorithm
        '''
        ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

        # Draw the track window on the frame
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return

# 2. CamShift
def CamShift (cap):

    # Read the first frame
    ret, frame = cap.read()
    '''
        WARNING! click ENTER
    '''
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Initialize the tracker
    roi = frame[y:y + h, x:x + w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 256], 1)
        '''
        Apply the CamShift algorithm
        '''
        # Apply the CamShift algorithm
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        # Draw the track window on the frame
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


# 3. CamShift_2
def CamShift_2 (cap):

    '''
    Трекер зображення, яке виділяється мишою на потоковому відео
    алгоритм ідентифікації: "Відстеження кольорових просторів":
     - виділяється область кадру відеопотоку;
     - зображення в області кадра визначається сукупністю точок (растрів) із відповідним кольорами: "маска зображення - образ";
     - в наступному кадрі, з урахуванням геометричних спотворень, визначається область кадра наближена до "маски кадра".
     - процес повторюється в межах відеопотоку - статичних зображень, що змінюються із заданою частотою.

    Prateek Joshi Artificial Intelligence applications with Python, Part 13:
    https://mazz.keybase.pub/ebooks/ai/9781786464392-ARTIFICIAL_INTELLIGENCE_WITH_PYTHON.pdf
    Scripts:
    https://github.com/PacktPublishing/Artificial-Intelligence-with-Python

    '''

    # Define a class to handle object tracking related functionality
    class ObjectTracker(object):
        def __init__(self, scaling_factor=0.5):

            # Initialize the video capture object from the webcam
            # self.cap = cv2.VideoCapture(0)

            # Initialize the video capture object
            self.cap = cap

            # Capture the frame from the webcam
            _, self.frame = self.cap.read()

            # Scaling factor for the captured frame
            self.scaling_factor = scaling_factor

            # Resize the frame
            self.frame = cv2.resize(self.frame, None,
                                    fx=self.scaling_factor, fy=self.scaling_factor,
                                    interpolation=cv2.INTER_AREA)

            # Create a window to display the frame
            cv2.namedWindow('Object Tracker')

            # Set the mouse callback function to track the mouse
            cv2.setMouseCallback('Object Tracker', self.mouse_event)

            # Initialize variable related to rectangular region selection
            self.selection = None

            # Initialize variable related to starting position
            self.drag_start = None

            # Initialize variable related to the state of tracking
            self.tracking_state = 0

        # Define a method to track the mouse events
        def mouse_event(self, event, x, y, flags, param):
            # Convert x and y coordinates into 16-bit numpy integers
            x, y = np.int16([x, y])

            # Check if a mouse button down event has occurred
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.tracking_state = 0

            # Check if the user has started selecting the region
            if self.drag_start:
                if flags & cv2.EVENT_FLAG_LBUTTON:
                    # Extract the dimensions of the frame
                    h, w = self.frame.shape[:2]

                    # Get the initial position
                    xi, yi = self.drag_start

                    # Get the max and min values
                    x0, y0 = np.maximum(0, np.minimum([xi, yi], [x, y]))
                    x1, y1 = np.minimum([w, h], np.maximum([xi, yi], [x, y]))

                    # Reset the selection variable
                    self.selection = None

                    # Finalize the rectangular selection
                    if x1 - x0 > 0 and y1 - y0 > 0:
                        self.selection = (x0, y0, x1, y1)

                else:
                    # If the selection is done, start tracking
                    self.drag_start = None
                    if self.selection is not None:
                        self.tracking_state = 1

        # Method to start tracking the object
        def start_tracking(self):
            # Iterate until the user presses the Esc key
            while True:
                # Capture the frame from webcam
                _, self.frame = self.cap.read()

                # Resize the input frame
                self.frame = cv2.resize(self.frame, None,
                                        fx=self.scaling_factor, fy=self.scaling_factor,
                                        interpolation=cv2.INTER_AREA)

                # Create a copy of the frame
                vis = self.frame.copy()

                # Convert the frame to HSV colorspace
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

                '''
                Create the mask based on predefined thresholds !!!! -> object detection and tracking !!!
                '''
                mask = cv2.inRange(hsv, np.array((0., 60., 32.)),
                                   np.array((180., 255., 255.)))

                # Check if the user has selected the region
                if self.selection:
                    # Extract the coordinates of the selected rectangle
                    x0, y0, x1, y1 = self.selection

                    # Extract the tracking window
                    self.track_window = (x0, y0, x1 - x0, y1 - y0)

                    # Extract the regions of interest
                    hsv_roi = hsv[y0:y1, x0:x1]
                    mask_roi = mask[y0:y1, x0:x1]

                    # Compute the histogram of the region of
                    # interest in the HSV image using the mask
                    hist = cv2.calcHist([hsv_roi], [0], mask_roi,
                                        [16], [0, 180])

                    # Normalize and reshape the histogram
                    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                    self.hist = hist.reshape(-1)

                    # Extract the region of interest from the frame
                    vis_roi = vis[y0:y1, x0:x1]

                    # Compute the image negative (for display only)
                    cv2.bitwise_not(vis_roi, vis_roi)
                    vis[mask == 0] = 0

                # Check if the system in the "tracking" mode
                if self.tracking_state == 1:
                    # Reset the selection variable
                    self.selection = None

                    # Compute the histogram back projection
                    hsv_backproj = cv2.calcBackProject([hsv], [0],
                                                       self.hist, [0, 180], 1)

                    # Compute bitwise AND between histogram
                    # backprojection and the mask
                    hsv_backproj &= mask

                    # Define termination criteria for the tracker
                    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                 10, 1)

                    # Apply CAMShift on 'hsv_backproj'
                    track_box, self.track_window = cv2.CamShift(hsv_backproj,
                                                                self.track_window, term_crit)

                    # Draw an ellipse around the object
                    cv2.ellipse(vis, track_box, (0, 255, 0), 2)

                # Show the output live video
                cv2.imshow('Object Tracker', vis)

                # Stop if the user hits the 'Esc' key
                c = cv2.waitKey(5)
                if c == 27:
                    break

            # Close all the windows
            cv2.destroyAllWindows()

    ObjectTracker().start_tracking()

    return

# -------------------------- Головні виклики -----------------------------------
if __name__ == '__main__':

    # Read the video
    cap = cv2.VideoCapture('V_4.mp4')
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('V_6.mp4')

    print('Оберіть метод object tracking :')
    print('1 - MeanShift')
    print('2 - CamShift')
    print('3 - CamShift_2')

    mode = int(input('mode:'))

    if (mode == 1):
        print('1 - MeanShift')
        MeanShift(cap)

    if (mode == 2):
        print('2. CamShift')
        CamShift (cap)

    if (mode == 3):
        print('3. CamShift_2')
        CamShift_2 (cap)



