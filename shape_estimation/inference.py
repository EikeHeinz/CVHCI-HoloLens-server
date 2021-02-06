#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image_object
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from pytorch3d.io import generate_obj
from pytorch3d.structures import Meshes
from PIL import Image

# required so that .register() calls are executed in module scope
import meshrcnn.data  # noqa
import meshrcnn.modeling  # noqa
import meshrcnn.utils  # noqa
from meshrcnn.config import get_meshrcnn_cfg_defaults
from meshrcnn.evaluation import transform_meshes_to_camera_coord_system

import cv2


WEIGHTS_PATH = os.path.abspath('./shape_estimation/models/meshrcnn_R50.pth')
CONFIG_FILE = os.path.abspath('./shape_estimation/models/meshrcnn_R50_FPN.yaml')
FOCAL_LENGTH = 10.0
DETECT_THRESH = 0.9

class ShapeEstimationModel(object):
    def __init__(self, input_image, rois, weight_path=WEIGHTS_PATH, cfg_file=CONFIG_FILE, focal_length=FOCAL_LENGTH, detect_thresh= DETECT_THRESH):
        """
        Args:
            input_image: PIL image object
            roi: (y1, x1, y2, x2) detection bounding boxes
        """
        self.input_image = input_image
        self.rois = rois
        self.weight_path = weight_path
        self.focal_length = focal_length
        self.detect_thresh = detect_thresh

        self.cfg = self.setup_cfg(cfg_file)
        self.objects = []

    def setup_cfg(self, cfg_file):
        cfg = get_cfg()
        get_meshrcnn_cfg_defaults(cfg)
        cfg.merge_from_file(cfg_file)
        # model weights path
        cfg.merge_from_list(['MODEL.WEIGHTS', self.weight_path])
        cfg.freeze()
        return cfg

    def visualize_image(self, image):
        demo = VisualizationDemo(
            self.cfg, self.focal_length
        )
        # use PIL, to be consistent with evaluation
        img = read_image_object(image)
        predictions = demo.run_on_image(img, focal_length=self.focal_length)
        object = demo.object
        return object

    def get_detections(self):
        for roi in self.rois:
            roi_image = self.input_image.crop((roi[1], roi[0], roi[3], roi[2]))
            model_object = self.visualize_image(roi_image)
            self.objects.append(model_object)
        return self.objects


class VisualizationDemo(object):

    def __init__(self, cfg, focal_length, vis_highest_scoring=True):
        """
        Args:
            cfg (CfgNode):
            vis_highest_scoring (bool): If set to True visualizes only
                                        the highest scoring prediction
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.colors = self.metadata.thing_colors
        self.cat_names = self.metadata.thing_classes

        self.cpu_device = torch.device("cpu")
        self.vis_highest_scoring = vis_highest_scoring
        self.predictor = DefaultPredictor(cfg)
        self.object = ""

    def run_on_image(self, image, focal_length=10.0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            focal_length (float): the focal_length of the image

        Returns:
            predictions (dict): the output of the model.
        """
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        # camera matrix
        imsize = [image.shape[0], image.shape[1]]
        # focal <- focal * image_width / 32
        focal_length = image.shape[1] / 32 * focal_length
        K = [focal_length, image.shape[1] / 2, image.shape[0] / 2]

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            scores = instances.scores
            boxes = instances.pred_boxes
            labels = instances.pred_classes
            masks = instances.pred_masks
            meshes = Meshes(
                verts=[mesh[0] for mesh in instances.pred_meshes],
                faces=[mesh[1] for mesh in instances.pred_meshes],
            )
            pred_dz = instances.pred_dz[:, 0] * (boxes.tensor[:, 3] - boxes.tensor[:, 1])
            tc = pred_dz.abs().max() + 1.0
            zranges = torch.stack(
                [
                    torch.stack(
                        [
                            tc - tc * pred_dz[i] / 2.0 / focal_length,
                            tc + tc * pred_dz[i] / 2.0 / focal_length,
                        ]
                    )
                    for i in range(len(meshes))
                ],
                dim=0,
            )

            Ks = torch.tensor(K).to(self.cpu_device).view(1, 3).expand(len(meshes), 3)
            meshes = transform_meshes_to_camera_coord_system(
                meshes, boxes.tensor, zranges, Ks, imsize
            )

            if self.vis_highest_scoring:
                det_ids = [scores.argmax().item()]
            else:
                det_ids = range(len(scores))

            for det_id in det_ids:
                self.visualize_prediction(
                    det_id,
                    image,
                    boxes.tensor[det_id],
                    labels[det_id],
                    scores[det_id],
                    masks[det_id],
                    meshes[det_id],
                )

        return predictions

    def visualize_prediction(
        self, det_id, image, box, label, score, mask, mesh, alpha=0.6, dpi=200
    ):

        mask_color = np.array(self.colors[label], dtype=np.float32)
        cat_name = self.cat_names[label]
        thickness = max([int(np.ceil(0.001 * image.shape[0])), 1])
        box_color = (0, 255, 0)  # '#00ff00', green
        text_color = (218, 227, 218)  # gray

        composite = image.copy().astype(np.float32)

        # overlay mask
        idx = mask.nonzero()
        composite[idx[:, 0], idx[:, 1], :] *= 1.0 - alpha
        composite[idx[:, 0], idx[:, 1], :] += alpha * mask_color

        # overlay box
        (x0, y0, x1, y1) = (int(x + 0.5) for x in box)
        composite = cv2.rectangle(
            composite, (x0, y0), (x1, y1), color=box_color, thickness=thickness
        )
        composite = composite.astype(np.uint8)

        # overlay text
        font_scale = 0.001 * image.shape[0]
        font_thickness = thickness
        font = cv2.FONT_HERSHEY_TRIPLEX
        text = "%s %.3f" % (cat_name, score)
        ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, font_thickness)
        # Place text background.
        if x0 + text_w > composite.shape[1]:
            x0 = composite.shape[1] - text_w
        if y0 - int(1.2 * text_h) < 0:
            y0 = int(1.2 * text_h)
        back_topleft = x0, y0 - int(1.3 * text_h)
        back_bottomright = x0 + text_w, y0
        cv2.rectangle(composite, back_topleft, back_bottomright, box_color, -1)
        # Show text
        text_bottomleft = x0, y0 - int(0.2 * text_h)
        cv2.putText(
            composite,
            text,
            text_bottomleft,
            font,
            font_scale,
            text_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

        verts, faces = mesh.get_mesh_verts_faces(0)
        self.object = generate_obj(verts, faces)


def get_parser():
    parser = argparse.ArgumentParser(description="Shape Estimation Demo")
    parser.add_argument(
        '-cfg',
        "--config-file",
        default = CONFIG_FILE,
        metavar = "FILE",
        help = "path to network config file"
    )
    parser.add_argument(
        '-img',
        "--image",
        default = True,
        help = "A path to an input image"
    )
    parser.add_argument(
        '-f',
        "--focal-length",
        type = float,
        default = 20.0,
        help = "Focal length for the image"
    )
    parser.add_argument(
        "--onlyhighest",
        action = "store_true",
        default = True,
        help = "will return only the highest scoring detection"
    )
    parser.add_argument(
        '-roi',
        '--roi',
        help = "(y1, x1, y2, x2) detection bounding boxes",
        type = str
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    """
    Parameters:
    --image: path of image
    --roi: "y1, x1, y2, x2"
    example:
    python shape_estimation/inference.py --image ./shape_estimation/0007.png --roi "95, 25, 390, 470"

    """

    args = get_parser().parse_args()

    im_original = Image.open('./shape_estimation/images/0007.png')

    roi_list = [
        [80, 20, 390, 475]
    ]

    shape_estimation_model = ShapeEstimationModel(im_original, roi_list)
    shape_objects = shape_estimation_model.get_detections()
    for x in shape_objects:
        print(x)
