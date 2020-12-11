import os

import socket
import numpy as np
import skimage

from instance_segmentation.inference import MaskRCNNInference, WEIGHTS_PATH, IMAGE_DIR
# import sys


def main():
    print("Setting up Socket")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 10000)
    print("starting socket on %s port %s" % server_address)
    sock.bind(server_address)

    # Listen for connections
    sock.listen()
    print("Listening for connections...")

    # initialize image variables
    image_name = os.listdir(IMAGE_DIR)[0]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

    depth_image = np.zeros((1, 1))

    # this is the point where the cursor was during the select gesture
    image_point = (-1, -1)

    # Initialize the inference model 
    seg_inference_model = MaskRCNNInference(WEIGHTS_PATH)
    sample_detections = seg_inference_model.get_detections([image])[0]

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
                    break
        finally:
            connection.close()


def estimate_shape(cropped_image):
    # TODO call shape estimation
    print("shape estimation not yet implemented")


if __name__ == '__main__':
    main()
