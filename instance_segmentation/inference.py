import os

import PIL
import skimage

from instance_segmentation.mrcnn.model import MaskRCNN
from instance_segmentation.mrcnn.sun import SunConfig, CLASSES
from instance_segmentation.mrcnn import visualize
from instance_segmentation.mrcnn.visualize import display_images

INFERENCE_CLASSES = ['BG']
INFERENCE_CLASSES.extend(CLASSES)

IMAGE_DIR = os.path.abspath('./instance_segmentation/images')
WEIGHTS_PATH = os.path.abspath('./instance_segmentation/models/rgb_mask_rcnn.h5')

class MaskRCNNInference(MaskRCNN):
    """ CVHCI inference model for MaskRCNN matterport implementation """

    def __init__(self, detect_thresh: float = 0.65, weights_path: str = None):
        """ CVHCI Inference model for MaskRCNN matterport implementation
        Arguments: 
            :param detect_thresh: Threshold used for detecting images.
            :param weights_path: Path to weights for Mask RCNN
        """
        class InferenceConfig(SunConfig):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = detect_thresh

        config = InferenceConfig()

        super().__init__(model_dir = './', mode='inference', config=config)

        if weights_path is not None: 
            self.load_weights(weights_path, by_name=True)

        else: 
            print('No weights loaded, consider to provide weights path.')

    def get_detections(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """        
        # Custom Transformations on the output are done here.
        return self.detect(images)

if __name__ == '__main__':
    # Load a random image from the images folder
    file_names = os.listdir(IMAGE_DIR)
    for image_name in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

        model = MaskRCNNInference(weights_path=WEIGHTS_PATH)

        # Run detection
        results = model.detect([image])

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    INFERENCE_CLASSES, r['scores'],
                                    save_path=os.path.join(IMAGE_DIR, f'{image_name[:-4]}_detected.png'))
