from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from misc_utils import parse_anchors, read_class_names
from nms_utils import gpu_nms
from plot_utils import get_color_table, plot_one_box
from data_utils import letterbox_resize

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--anchor_path", type=str, default="../../data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="../../data/classes.txt",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="../../../../data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
parser.add_argument("--pb_path", type=str, default="../../data/pb",
					help="The directory of pb files")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
	
    saver.save(sess, args.pb_path+'/pb.ckpt')
    tf.io.write_graph(sess.graph_def, args.pb_path, 'model.pb', as_text=False)
    tf.io.write_graph(sess.graph_def, args.pb_path, 'model.pbtxt', as_text=True)