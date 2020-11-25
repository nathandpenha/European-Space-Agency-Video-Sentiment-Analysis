#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import dlib
import math
import imutils
from openvino.inference_engine import IECore
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_eye = np.array([36, 37, 38, 39, 40, 41])
right_eye = np.array([42, 43, 44, 45, 46, 47])


# %%

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # find length of triangle edges
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = imutils.rotate(img, direction * angle)

    return img  # return img anyway


# %%

def resize(image, percent):
    width = int(image.shape[1] * percent)
    height = int(image.shape[0] * percent)

    dsize = (width, height)
    image = cv2.resize(image, dsize)

    return image


# %%

def spatial_normalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 1)

    if len(detections) > 0:
        detected_face = detections[0]
        face_shape = predictor(gray, detected_face)
        landmarks = shape_to_np(face_shape)

        left_eye_center = np.mean(landmarks[left_eye], axis=0)
        right_eye_center = np.mean(landmarks[right_eye], axis=0)
        image = alignment_procedure(image, left_eye_center, right_eye_center)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector(gray, 1)
        if len(detections) > 0:
            detected_face = detections[0]
            face_shape = predictor(gray, detected_face)
            landmarks = shape_to_np(face_shape)

            left_eye_center = np.mean(landmarks[left_eye], axis=0)
            right_eye_center = np.mean(landmarks[right_eye], axis=0)

            inner_eyes = (right_eye_center[0] - left_eye_center[0]) / 2
            w = inner_eyes * 2.4
            h = inner_eyes * 4.5
            x = left_eye_center[0] + ((right_eye_center[0] - left_eye_center[0]) / 2) - (w / 2)
            y = right_eye_center[1] - (inner_eyes * 1.3)

            image = image[int(y):int(y + h), int(x):int(x + w)]

    return image


# %%

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_cascade4 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detect_only(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if (len(faces) == 0 or len(faces) > 1):
        faces = face_cascade1.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if (len(faces) == 0 or len(faces) > 1):
            faces = face_cascade2.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if (len(faces) == 0 or len(faces) > 1):
                faces = face_cascade3.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if (len(faces) == 0 or len(faces) > 1):
                    faces = face_cascade4.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if (len(faces) == 1):
        for (x, y, w, h) in faces:
            image = image[y:y + h, x:x + w]
    else:
        detections = detector(gray, 1)

        if len(detections) > 0:
            d = detections[0]
            left = d.left();
            right = d.right()
            top = d.top();
            bottom = d.bottom()
            image = image[top:bottom, left:right]

    return image


def face_detect_align(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 1)

    if len(detections) > 0:
        detected_face = detections[0]
        img_shape = predictor(image, detected_face)
        image = dlib.get_face_chip(image, img_shape, size=image.shape[0], padding=0.00)

    return image


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    print(args.input)
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    # Read and pre-process input images
    print("shape")
    print(net.input_info[input_blob].input_data.shape)
    n, c, h, w, d = net.input_info[input_blob].input_data.shape
    videos = np.ndarray(shape=(n, c, h, w, d))
    for j in range(n):
        cap = cv2.VideoCapture(args.input[j])
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frames = [x * nframe / d for x in range(d)]

        start_preprocess = time.time()
        framearray = []

        for i in range(d):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            #         frame = cv2.resize(frame, (height, width))

            frame = resize(frame, 0.3)
            frame = spatial_normalization(frame)
            #         frame = face_detect_only(frame)
            #         frame = face_detect_align(frame)
            frame = cv2.resize(frame, (32, 32))

            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        framearray = np.array(framearray)
        print(framearray.shape)

        framearray = np.repeat(framearray[..., np.newaxis], 1, -1)
        framearray = framearray.transpose((3, 1, 2, 0))
        framearray = framearray / 255.  # .astype('float32')
        print(framearray.shape)
        end_preprocess = time.time()
        print("Preprocessing Performance")
        print(end_preprocess - start_preprocess)

        # video = cv2.imread(args.input[i])
        # print(video.shape)
        # if image.shape[:-1] != (h, w):
        #     log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
        #     image = cv2.resize(image, (w, h))
        # image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        videos[j] = framearray
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    #
    # Start sync inference
    log.info("Starting inference in synchronous mode")
    start_inference = time.time()
    res = exec_net.infer(inputs={input_blob: videos})
    end_inference = time.time()
    print("Inference Performance")
    print(end_inference - start_inference)
    #res = exec_net.infer({'data': args.input})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("Top {} results: ".format(args.number_top))
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image {}\n".format(args.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n")
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")

if __name__ == '__main__':
    sys.exit(main() or 0)