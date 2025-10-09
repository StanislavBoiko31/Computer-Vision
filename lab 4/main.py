import cv2
import numpy as np
from enum import Enum

class TrackingMethod(Enum):    
    MEAN_SHIFT = 1
    CAM_SHIFT = 2
    CSRT = 3

class TrackerState:
    def __init__(self):
        self.tracking = False
        self.bbox = None
        self.tracker = None
        self.method = None
        self.roi_hist = None
        self.track_window = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.paused = False
        self.current_frame = None

def select_roi(frame, window_name):
    bbox = cv2.selectROI(window_name, frame, False, False)
    cv2.destroyWindow(window_name)
    return bbox

def process_key_actions(state):
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  
        return False
    elif key == ord('p'):  
        state.paused = not state.paused
    elif key == ord('r'):  
        state.tracking = False
    return True

def mean_shift_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    state = TrackerState()
    state.method = 'meanshift'
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.tracking = False
    
    cv2.namedWindow('MeanShift Tracking')
    cv2.setMouseCallback('MeanShift Tracking', on_mouse)
    
    print("\nControls:")
    print("- Press 'p' to pause/resume")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")
    print("- Click to select new object")
    
    while True:
        if not state.paused:
            ret, frame = cap.read()
            if not ret:
                break
            state.current_frame = frame.copy()
        else:
            frame = state.current_frame.copy()
            
        if not state.tracking:
            cv2.putText(frame, "Select object to track", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('MeanShift Tracking', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                state.paused = not state.paused
                continue
                
            bbox = select_roi(frame, 'MeanShift Tracking')
            if bbox[2] > 0 and bbox[3] > 0:  
                x, y, w, h = [int(v) for v in bbox]
                state.track_window = (x, y, w, h)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                state.roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv2.normalize(state.roi_hist, state.roi_hist, 0, 255, cv2.NORM_MINMAX)
                state.tracking = True
            else:
                continue
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], state.roi_hist, [0, 180], 1)
        
        ret, state.track_window = cv2.meanShift(dst, state.track_window, state.term_crit)
        
        x, y, w, h = state.track_window
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        status = "Paused" if state.paused else "Tracking"
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('MeanShift Tracking', frame)
        
        if not process_key_actions(state):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def kcf_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    state = TrackerState()
    state.method = 'kcf'
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.tracking = False
    
    cv2.namedWindow('KCF Tracking')
    cv2.setMouseCallback('KCF Tracking', on_mouse)
    
    print("\nControls:")
    print("- Press 'p' to pause/resume")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")
    print("- Click to select new object")
    
    while True:
        if not state.paused:
            ret, frame = cap.read()
            if not ret:
                break
            state.current_frame = frame.copy()
        else:
            frame = state.current_frame.copy()
            
        if not state.tracking:
            cv2.putText(frame, "Select object to track", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('KCF Tracking', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                state.paused = not state.paused
                continue
                
            bbox = select_roi(frame, 'KCF Tracking')
            if bbox[2] > 0 and bbox[3] > 0: 
                state.tracker = cv2.TrackerKCF_create()
                state.tracker.init(frame, tuple(bbox))
                state.tracking = True
            else:
                continue
        
        success, bbox = state.tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        status = "Paused" if state.paused else "Tracking"
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('KCF Tracking', frame)
        
        if not process_key_actions(state):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def csrt_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    state = TrackerState()
    state.method = 'csrt'
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.tracking = False
    
    cv2.namedWindow('CSRT Tracking')
    cv2.setMouseCallback('CSRT Tracking', on_mouse)
    
    print("\nControls:")
    print("- Press 'p' to pause/resume")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")
    print("- Click to select new object")
    
    while True:
        if not state.paused:
            ret, frame = cap.read()
            if not ret:
                break
            state.current_frame = frame.copy()
        else:
            frame = state.current_frame.copy()
            
        if not state.tracking:
            cv2.putText(frame, "Select object to track", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('CSRT Tracking', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                state.paused = not state.paused
                continue
                
            bbox = select_roi(frame, 'CSRT Tracking')
            if bbox[2] > 0 and bbox[3] > 0: 
                state.tracker = cv2.TrackerCSRT_create()
                state.tracker.init(frame, tuple(bbox))
                state.tracking = True
            else:
                continue
        
        success, bbox = state.tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        status = "Paused" if state.paused else "Tracking"
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('CSRT Tracking', frame)
        
        if not process_key_actions(state):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'video.mp4'
    
    print("Select tracking method:")
    print("1. MeanShift")
    print("2. KCF")
    print("3. CSRT")
    
    choice = input("Enter your choice (1-3): ")
    
    try:
        method = int(choice)
        if method < 1 or method > 3:
            raise ValueError("Invalid choice")
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 3.")
        return
    
    if method == TrackingMethod.MEAN_SHIFT.value:
        mean_shift_tracking(video_path)
    elif method == TrackingMethod.CAM_SHIFT.value:
        kcf_tracking(video_path)
    elif method == TrackingMethod.CSRT.value:
        csrt_tracking(video_path)

if __name__ == "__main__":
    main()
