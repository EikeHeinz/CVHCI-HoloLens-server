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
    
    def __init__(self, detect_thresh=0.65, weights_path=None):

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

    def get_detections(self, image):
        
        if type(image) is not list:
            return self.detect([image])
        else: 
            return self.detect(image)

if __name__ == '__main__':
    # Load a random image from the images folder
    file_names = os.listdir(IMAGE_DIR)
    for image_name in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

        model = MaskRCNNInference(weights_path=WEIGHTS_PATH)

        # Run detection
        results = model.get_detections([image])

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    INFERENCE_CLASSES, r['scores'],
                                    save_path=os.path.join(IMAGE_DIR, f'{image_name[:-4]}_detected.png'))
