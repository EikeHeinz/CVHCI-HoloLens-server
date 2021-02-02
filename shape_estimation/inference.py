#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import numpy as np
import os
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from PIL import Image

# required so that .register() calls are executed in module scope
import meshrcnn.data  # noqa
import meshrcnn.modeling  # noqa
import meshrcnn.utils  # noqa
from meshrcnn.config import get_meshrcnn_cfg_defaults
from meshrcnn.evaluation import transform_meshes_to_camera_coord_system

import cv2


IMAGE_DIR = os.path.abspath('./shape_estimation/images')
OUTPUT_DIR = os.path.abspath('./shape_estimation/output_objects')

WEIGHTS_PATH = os.path.abspath('./shape_estimation/models/meshrcnn_R50.pth')
CONFIG_FILE = os.path.abspath('./shape_estimation/models/meshrcnn_R50_FPN.yaml')


class VisualizationDemo(object):

    object_full_path = ''
    def __init__(self, cfg, vis_highest_scoring=True, output_dir="./vis"):
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

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

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

        save_file = os.path.join(self.output_dir, "%d_mask_%s_%.3f.png" % (det_id, cat_name, score))
        cv2.imwrite(save_file, composite[:, :, ::-1])

        save_file = os.path.join(self.output_dir, "%d_mesh_%s_%.3f.obj" % (det_id, cat_name, score))
        verts, faces = mesh.get_mesh_verts_faces(0)
        self.object_full_path = save_file
        save_obj(save_file, verts, faces)


def visualize_image(args, cfg, image_name, image_path):
    demo = VisualizationDemo(
        cfg, vis_highest_scoring = args.onlyhighest, output_dir = os.path.join(args.output, image_name)
    )
    # use PIL, to be consistent with evaluation
    img = read_image(image_path, format="BGR")
    predictions = demo.run_on_image(img, focal_length=args.focal_length)

    obj_full_path = demo.object_full_path
    return obj_full_path


def setup_cfg(args):
    cfg = get_cfg()
    get_meshrcnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # model weights path
    cfg.merge_from_list(['MODEL.WEIGHTS', WEIGHTS_PATH])
    cfg.freeze()
    return cfg


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
        help = "A path to an input image",
        required = True
    )
    parser.add_argument(
        '-input',
        "--input-dir",
        default = IMAGE_DIR,
        help = "A path to the input images directory"
    )
    parser.add_argument(
        '-output',
        "--output",
        default = OUTPUT_DIR,
        help = "A directory to save output visualizations"
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
        required = True,
        type = str
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


## INPUT: original_image_path, ROI(Array)
def crop_rois(image_full_path, roi):
    """
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    Set the cropping area with box=(left, upper, right, lower) = (x1, y1, x2, y2)
    """
    # crop_box = [roi[1], roi[0], roi[3], roi[2]]
    image_name = image_full_path.split("/")[-1].split(".")[0]
    image_ext = image_full_path.split("/")[-1].split(".")[1]

    im_original = Image.open(image_full_path)
    im_crop = im_original.crop((roi[1], roi[0], roi[3], roi[2]))
    roi_img_name = str(roi[1]) + '-' + str(roi[0]) + '-' + str(roi[3]) + '-' + str(roi[2]) + '_' + image_name
    roi_img_full_name = roi_img_name + '.' + image_ext
    roi_img_save_dir = args.output + '/roi_images'
    os.makedirs(roi_img_save_dir, exist_ok=True)
    roi_img_full_path = roi_img_save_dir + '/' + roi_img_full_name
    im_crop.save(roi_img_full_path, quality=95)

    return roi_img_full_path, roi_img_name


if __name__ == "__main__":
    """
    Parameters:
    --image: path of image
    --roi: "y1, x1, y2, x2"
    example:
    python shape_estimation/inference.py --image ./shape_estimation/0007.png --roi "95, 25, 390, 470"

    Output: 'object_full_path', path of generated object
    """
    mp.set_start_method("spawn", force=True)

    args = get_parser().parse_args()

    cfg = setup_cfg(args)

    image_full_path = args.image

    roi_list = [int(item) for item in args.roi.split(',')]

    roi_image_full_path, roi_image_name = crop_rois(image_full_path, roi_list)

    object_full_path =  visualize_image(args, cfg, roi_image_name, roi_image_full_path)
