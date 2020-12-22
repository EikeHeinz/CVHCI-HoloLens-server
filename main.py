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

    obj = load_obj_from_file()

    # initialize image variables
    # image_name = os.listdir(IMAGE_DIR)[0]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

    # depth_image = np.zeros((1, 1))

    # this is the point where the cursor was during the select gesture
    # image_point = (-1, -1)

    # Initialize the inference model 
    # seg_inference_model = MaskRCNNInference(weights_path=WEIGHTS_PATH)
    # sample_detections = seg_inference_model.get_detections([image])[0]

    # plt.imshow(sample_detections['masks'][:, :, 0])
    # plt.show()

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
            buffer = ""
            extracted_size = False
            expected_size = 1024
            actual_size = 0

            print("%s connected" % client_address[0])

            while actual_size < expected_size:
                data = connection.recv(1024)
                print("Received %s" % data)
                if data:
                    buffer = buffer + data.decode("UTF-8")
                    actual_size += len(data)
                    print("got more data: %s/%s" % (actual_size, expected_size))
                    if not extracted_size:
                        initial_values = buffer
                        message_part = initial_values.split(";")
                        message_length = message_part[0].strip()
                        expected_size = int(message_length)
                        print("expecting: %s bytes" % expected_size)
                        extracted_size = True

            print("no more data received")
            print("total received: %s" % buffer)
            # decode_message(buffer)
            # This is entrypoint for model pipeline
            connection.sendall(obj.encode("UTF-8"))

            print("Listening for connections...")

        finally:
            connection.close()


def load_obj_from_file():
    filepath = 'model2.obj'
    with open(filepath, 'r') as file:
        data = file.read()#.replace('\n', ' ')
        return data


def decode_message(message):
    # message is string containing identifiers for each part image (i), depth (d) and point (p)
    # message example: "i[0 1 2 3][4 5 6 7]; d [1][2]; p 0 1;" for image with 2 pixels
    data = message.split(";")
    point_begin_index = data[2].index("p") + 2
    point = data[2][point_begin_index:].strip()
    coordinates = tuple(map(float, point.split(" ")))
    print(str(coordinates))
    print("Not yet implemented")


def crop_image_to_area_round_point(image, image_point):
    print("cropping is not yet implemented")


def estimate_shape(cropped_image):
    # TODO call shape estimation
    print("shape estimation not yet implemented")


if __name__ == '__main__':
    main()
