import os

import socket
import numpy as np
import skimage

import matplotlib.pyplot as plt

import torch
import tensorflow as tf

from instance_segmentation.inference import MaskRCNNInference, WEIGHTS_PATH, IMAGE_DIR
# import sys


def main():
    print("PyTorch CUDA: ", torch.cuda.is_available())
    print("Tensorflow version: ", tf.__version__)

    # initialize image variables
    image_name = os.listdir(IMAGE_DIR)[0]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

    depth_image = np.zeros((1, 1))

    # this is the point where the cursor was during the select gesture
    image_point = (-1, -1)

    # Initialize the inference model 
    seg_inference_model = MaskRCNNInference(weights_path=WEIGHTS_PATH)
    sample_detections = seg_inference_model.get_detections([image])[0]

    plt.imshow(sample_detections['masks'][:, :, 0])
    plt.show()

    print("Setting up Socket")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 10000)
    print("starting socket on %s port %s" % server_address)
    sock.bind(server_address)

    # Listen for connections
    sock.listen()
    print("Listening for connections...")

    while True:
        # Wait for connection
        connection, client_address = sock.accept()

        try:
            print("%s connected" % client_address)
            while True:
                data = connection.recv(1024)
                print("Received %s" % data)
                if data:
                    print("got more data")
                else:
                    print("no more data received")
                    # This is entrypoint for model pipeline
                    # TODO how to know that all relevant data is present? we need image, depth_image and image_point
                    # decode message: i bytes d bytes p bytes
                    break
        finally:
            connection.close()


def decode_message(message):
    # message is string containing identifiers for each part image (i), depth (d) and point (p)
    # 
    print("Not yet implemented")

def estimate_shape(cropped_image):
    # TODO call shape estimation
    print("shape estimation not yet implemented")


if __name__ == '__main__':
    main()
