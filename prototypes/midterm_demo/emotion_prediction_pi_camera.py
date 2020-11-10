from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import logging as log
from openvino.inference_engine import IECore
import numpy as np

EMOTIONS = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprise']

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=False,
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


def prepare_image(image, net, input_blob):
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        images[i] = image
    return images


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    result = []
    face_image= None
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

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

    input_blob = next(iter(net.input_info))

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    image_size = (224, 224)

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    # allow the camera to warmup
    time.sleep(0.1)

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        # show the frame
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.3,
            minNeighbors=3
        )
        
        for (x, y, w, h) in faces:
            face_image = image[y:y + h, x:x + w]
        if face_image is not None:
          # resize an image for being able to use in a NN
          resized_face_image = cv2.resize(face_image, image_size)
          # general normalization
          resized_face_image = resized_face_image / 255
          images = prepare_image(resized_face_image, net, input_blob)
          res = exec_net.infer(inputs={input_blob: images})

          # cmd log with probability/emotion output
          for key, probs in res.items():
            result = probs        
            probs = np.squeeze(probs)
            index = (-probs).argsort()[:8]            
            log.info("Predicted probabality: {}".format(probs[index[0]]))
            log.info("Predicted emotion type: {}".format(EMOTIONS[index[0]]))

          # histogram drawing
          if result is not None:
            # White background over text
            cv2.rectangle(image, (0, 0),
                          (150, 330),
                          (0,0,0), -1)
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(image, emotion, (10, index * 40 + 40), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
                cv2.rectangle(image, (160, index * 40 + 20), (160 + int(result[0][index] * 100 *1.5), (index + 1) * 40 + 4),
                          (0,0,0), -1)
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    sys.exit(main() or 0)
