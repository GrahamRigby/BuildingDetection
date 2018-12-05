# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util

def CNTraining():
    sys.path.append("..")
    CNWidths = []
    for imgName in os.listdir("CNTraining"):
        cwd = os.getcwd()
        #Path to the CN Tower Detector
        MODELPATH = os.path.join(cwd,'CN_Model','frozen_inference_graph.pb')

        # Opens the CN Tower detector model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODELPATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        # Loads net variable names and net parameter definitions
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image 
        image = cv2.imread(os.path.join(cwd,'CNTraining',imgName))
        image_expanded = np.expand_dims(image, axis=0)

        # Runs the model on selected image and records bounding boxes and scores 
        # surrounding the detected object of interest (CN Tower)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #percieved width of the CN tower measured in pixels
        CNWidth = float(image.shape[1] * (boxes[0][0][1] - boxes[0][0][3]))
        CNWidths.append(CNWidth)
    return CNWidths

def AUTraining():
    sys.path.append("..")
    AUWidths = []
    for imgName in os.listdir("AUTraining"):
        cwd = os.getcwd()
        #Path to the Aura Tower Detector
        MODELPATH = os.path.join(cwd,'Aura_Model','frozen_inference_graph.pb')

        # Opens the Aura Tower detector model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODELPATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
        
        # Loads net variable names and net parameter definitions
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image 
        image = cv2.imread(os.path.join(cwd,'AUTraining',imgName))
        image_expanded = np.expand_dims(image, axis=0)

        # Runs the model on selected image and records bounding boxes and scores 
        # surrounding the detected object of interest (Aura Tower)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        
        #percieved width of the Aura tower measured in pixels
        AUWidth = float(image.shape[1] * (boxes[0][0][1] - boxes[0][0][3]))
        AUWidths.append(AUWidth)
    return AUWidths

def TowerDetection():
    sys.path.append("..")
    towerWidths = []
    cwd = os.getcwd()
    for imgName in os.listdir("TestImages"):
        #Path to the CN Tower Detector
        MODELPATH = os.path.join(cwd,'CN_Model','frozen_inference_graph.pb')

        # Opens the CN Tower detector model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODELPATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        # Loads net variable names and net parameter definitions
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image 
        image = cv2.imread(os.path.join(cwd,'TestImages',imgName))
        image_expanded = np.expand_dims(image, axis=0)

        # Runs the model on selected image and records bounding boxes and scores 
        # surrounding the detected object of interest (CN Tower)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #percieved width of the CN tower measured in pixels
        CNWidth = float(image.shape[1] * (boxes[0][0][1] - boxes[0][0][3]))
        cnStart = boxes[0][0][1]

        MODELPATH = os.path.join(cwd,'Aura_Model','frozen_inference_graph.pb')
        # Opens the Aura Tower detector model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODELPATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
        
        # Loads net variable names and net parameter definitions
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image 
        image = cv2.imread(os.path.join(cwd,'TestImages',imgName))
        image_expanded = np.expand_dims(image, axis=0)

        # Runs the model on selected image and records bounding boxes and scores 
        # surrounding the detected object of interest (Aura Tower)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        
        #percieved width of the Aura tower measured in pixels
        AUWidth = float(image.shape[1] * (boxes[0][0][1] - boxes[0][0][3]))
        auStart = boxes[0][0][1]
        orientation = 0 if(cnStart < auStart) else 1
        towerWidths.append([CNWidth, AUWidth, orientation])
    return towerWidths

if __name__ == "__main__":
    print("Done")