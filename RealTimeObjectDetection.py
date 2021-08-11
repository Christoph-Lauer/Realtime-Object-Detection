# @file         RealTimeObjectDetection.py
# @version      1.0
# @date         2021
# @author       Christoph Lauer
# @client       none
# @OS           macOS 12 Monterey
# @contributors none
# @biref        Realtime Camera Object-Extraction with pretrained YOLO Darknet Model 
# @shell        Python
# @requires     Python 3, OpenCV...
# @usage        python RealTimeObjectExtraction.py
# @arguments    none
# @see          https://pjreddie.com/darknet/yolo/
# @notes        none
# @todo         finished so far
# @copyright    Christoph Lauer Engineering
# @license      cle commercial license

import numpy as np
import argparse
import cv2
import os
import time

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')
                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)
    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    return boxes, confidences, classIDs, idxs

if __name__ == '__main__':
    confidence=0.5      # Minimum confidence for a box to be detected.
    threshold = 0.3     # Threshold for Non-Max Suppression
    brightness = 1

    # Get the labels
    labels = open('model/coco.names').read().strip().split('\n')
    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')
    # Get the ouput layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        

    ### CAPTURE LOOP FOR THE CAMERA
    cap = cv2.VideoCapture(0)
    fps = 0
    while cap.isOpened():
        start = time.time()
        ret, image = cap.read()

        if not ret:
            print('Video file finished.')
            break

        ### BRIGHTEN UP IMAGE
        #####################
        if (brightness != 1):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert it to hsv
            hsv[:,:,2] *= brightness
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #####################
            
        ### DETECT
        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)
        ### DRAW BOX BOX
        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
        ### TEXT
        h,w,l = image.shape
        cv2.putText(image, "fps:" + str(fps), (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        ### SHOW IMAGE
        cv2.imshow('YOLO Object Detection', image)
        ### Q Key for Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps = round(1 / (time.time() - start),2)
            
    cap.release()
cv2.destroyAllWindows()