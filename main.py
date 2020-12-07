import socket
import numpy as np
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
    image = np.zeros((1, 1, 1))
    depth_image = np.zeros((1, 1))

    # this is the point where the cursor was during the select gesture
    image_point = (-1, -1)

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


def instance_segmentation(image, depth_image):
    # TODO call instance segmentation model
    print("Instance segmentation not yet implemented")


def estimate_shape(cropped_image):
    # TODO call shape estimation
    print("shape estimation not yet implemented")


if __name__ == '__main__':
    main()
