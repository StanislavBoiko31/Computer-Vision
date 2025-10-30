import numpy as np
import cv2
import imutils
import time
from collections import deque, OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

def load_models():
    print("[INFO] Loading models...")
    
    prototxt = 'Lab_work_6/MobileNetSSD_deploy.prototxt.txt'
    model = 'Lab_work_6/MobileNetSSD_deploy.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    
    face_cascade = cv2.CascadeClassifier('Lab_work_6/haarcascade_frontalface_default.xml')
    
    return net, face_cascade

def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def detect_people(frame, net, min_confidence=0.2, nms_threshold=0.3):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    people_boxes = []
    confidences = []
    
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > min_confidence:
            idx = int(detections[0, 0, i, 1])
            
            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                people_boxes.append([startX, startY, endX, endY])
                confidences.append(float(confidence))
    
    # Apply non-maximum suppression
    if len(people_boxes) > 0:
        people_boxes = non_max_suppression_fast(np.array(people_boxes), nms_threshold)
    
    return frame, people_boxes

def detect_faces(frame, people_boxes, face_cascade):
    
    face_boxes = []
    
    for (startX, startY, endX, endY) in people_boxes:
        person_roi = frame[max(0, startY):min(endY, frame.shape[0]), 
                          max(0, startX):min(endX, frame.shape[1])]
        
        if person_roi.size == 0:
            continue
        
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)
        
        for (x, y, w, h) in faces:
            abs_x = startX + x
            abs_y = startY + y
            
            face_boxes.append((abs_x, abs_y, abs_x + w, abs_y + h))
            
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + w, abs_y + h),
                (255, 0, 0), 2)
            cv2.putText(frame, "Face", (abs_x, abs_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, face_boxes


def main():
    try:
        net, face_cascade = load_models()
    except IOError as e:
        print(f"[ERROR] {e}")
        return
    
    # Initialize centroid tracker
    ct = CentroidTracker(maxDisappeared=30)
    
    # Initialize frame counter and frame skip
    frame_count = 0
    FRAME_SKIP = 2  # Process every 3rd frame
    
    # Initialize face detection queue
    face_queue = deque(maxlen=5)
    
    video_path = 'video_lab_6.mp4'
    print(f"[INFO] Opening video stream: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open video stream")
        return
    
    # Get video properties for FPS calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 0.03
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[INFO] End of video stream")
            break
        
        frame_count += 1
        if frame_count % (FRAME_SKIP + 1) != 0:
            continue
            
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        frame = imutils.resize(frame, width=800)
        
        frame, people_boxes = detect_people(frame, net, min_confidence=0.4)
        
        rects = []
        for (startX, startY, endX, endY) in people_boxes:
            rects.append((startX, startY, endX, endY))
        
        objects = ct.update(rects)
        
        for (objectID, centroid) in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        if frame_count % 3 == 0:  # Process faces every 3rd frame
            _, face_boxes = detect_faces(frame, people_boxes, face_cascade)
            face_queue.append(face_boxes)
        
        if face_queue:
            face_boxes = face_queue[-1]
            for (startX, startY, endX, endY) in face_boxes:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        
        num_people = len(people_boxes)
        cv2.putText(frame, f"People: {num_people}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Enhanced People Identification", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  
            while True:
                key = cv2.waitKey(1) or 0xFF
                if key == ord('p') or key == ord('q'):
                    break
            if key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()