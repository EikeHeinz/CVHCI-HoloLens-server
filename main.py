import os
import socket
from timeit import default_timer as timer
import numpy as np
from numpy import array
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import torch
import tensorflow as tf
from instance_segmentation.inference import MaskRCNNInference, WEIGHTS_PATH, IMAGE_DIR
from instance_segmentation.mrcnn import visualize
from instance_segmentation.mrcnn.sun import CLASSES
from shape_estimation.inference import ShapeEstimationModel

INFERENCE_CLASSES = ['BG']
INFERENCE_CLASSES.extend(CLASSES)

# Set this accordingly
ipAddress = '127.0.0.1'
port = 10000

# Begin debug flags
use_existing_local_model = False
predict_existing_local_model = False

show_current_img = False
visualize_instances = False
# End debug flags


def main():
    print("PyTorch CUDA: ", torch.cuda.is_available())
    print("Tensorflow version: ", tf.__version__)

    seg_inference_model = MaskRCNNInference(weights_path=WEIGHTS_PATH)
    if use_existing_local_model:
        data = load_data_from_file()
        img, coordinates = decode_message(data)
        if show_current_img:
            plt.imshow(img)
            plt.show()

        if predict_existing_local_model:
            sample_detections = seg_inference_model.get_detections([img])[0]
            if visualize_instances:
                r = sample_detections
                visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                            INFERENCE_CLASSES, r['scores'],
                                            save_path=f'test_detected.png')

            pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
            rois = sample_detections['rois']
            shape_estimation = ShapeEstimationModel(pil_img, rois)
            obj_models = shape_estimation.get_detections()

            model_counter = 0
            for model in obj_models:
                write_model_to_file(model, model_counter)
                model_counter += 1

    # obj = load_obj_from_file()
    print("Setting up Socket")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (ipAddress, port)
    print("starting socket on %s port %s" % server_address)
    sock.bind(server_address)

    # Listen for connections
    sock.listen()
    print("Listening for connections...")

    while True:
        # Wait for connection
        connection, client_address = sock.accept()
        start_time = timer()

        try:
            buffer = ""
            extracted_size = False
            expected_size = 65536
            actual_size = 0
            timeout = 0

            print("%s connected" % client_address[0])

            while actual_size < expected_size:
                data = connection.recv(32768)
                print("Received %s" % len(data))
                if len(data) == 0:
                    timeout += 1
                if timeout == 10:
                    break
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
            print("total received: %s" % len(buffer))
            write_data_to_file(buffer)
            buffer = buffer[buffer.index(";") + 1:]
            #write_data_to_file(buffer)
            img, coordinates = decode_message(buffer)
            detections = seg_inference_model.get_detections([img])[0]
            rois = detections['rois']

            pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
            shape_estimation = ShapeEstimationModel(pil_img, rois)
            obj_models = shape_estimation.get_detections()

            # select correct model based on (x,y) tuple
            model_index = roi_index_of_point(coordinates[0], coordinates[1], rois)
            if model_index:
                print("point in an area, index is: " + str(model_index))
            else:
                print("point not in any area")
            model = obj_models[model_index]

            connection.sendall(model.encode("UTF-8"))
            end_time = timer()
            elapsed_time = end_time - start_time

            print("Total server processing time: %s" % elapsed_time)

            print("Listening for connections...")

        finally:
            connection.close()


def roi_index_of_point(x, y, rois):
    for roi in rois:
        if ((x >= roi[1] and x <= roi[3]) and (y >= roi[0] and y <= roi[2])):
            return rois.index(roi)
    return None


def decode_message(message):
    # message is string containing identifiers for each part image (i), depth (d) and point (p)
    # message example: "i[[[0,1,2],],[[3,4,5],],];d[[1],[2],];p[0,1];" for image with 2 pixels
    image_string = ""
    depth_string = ""
    point_string = ""

    data = message.split(";")
    for part in data:
        part = part.strip()
        if part.startswith("i"):
            image_string = part
        elif part.startswith("d"):
            depth_string = part
            # write_data_to_file(depth_string)
        elif part.startswith("p"):
            point_string = part
        elif len(part) == 0:
            print("empty part")
        else:
            print("invalid char: %s" % part[0])

    image_string = image_string[image_string.index("i") + 1:]
    image_string = "array(" + image_string + ", dtype=int)"
    rgb_img = eval(image_string.strip())
    print(rgb_img.shape)

    # depth_string = depth_string[depth_string.index("d") + 1:]
    # depth_string = "array(" + depth_string + ", dtype=int)"
    # depth_img = eval(depth_string.strip())

    point_begin_index = point_string.index("p") + 2
    point_end_index = point_string.index("]")
    point = point_string[point_begin_index:point_end_index].strip()
    coordinates = tuple(map(float, point.split(",")))
    print(str(coordinates))

    return rgb_img, coordinates


def load_obj_from_file():
    filepath = '0_mesh_chair_0.998.obj'
    with open(filepath, 'r') as file:
        data = file.read()
        return data


def write_data_to_file(data):
    filepath = 'paralleldata.txt'
    with open(filepath, 'w') as file:
        file.write(data)


def write_model_to_file(model, model_id):
    filepath = 'model' + str(model_id)+'.obj'
    with open(filepath, 'w') as file:
        file.write(model)


def load_data_from_file():
    filepath = 'paralleldata.txt'
    with open(filepath, 'r') as file:
        data = file.read()
        return data


if __name__ == '__main__':
    main()
