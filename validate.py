"""## Inference & evaluation using the trained model
Now, let's run inference with the trained model on the neuron validation dataset. First, let's create a predictor using
the model we just trained:
"""
import os
import random
from glob import glob

import cv2
import numpy as np
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer

from cfg import cfg
from train import neuron_metadata

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""

validation_images = glob("data/val/neurofinder*/images/*tiff")
for d in random.sample(validation_images, 3):
    example_im = cv2.imread(d)
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(example_im)
    v = Visualizer(
        example_im[:, :, ::-1],
        metadata=neuron_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. segmentation models only.
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(f"validation data {d}", example_im * 2 / np.max(example_im))
    cv2.imshow("predicted neurons", out.get_image())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""We can also evaluate its performance using AP metric implemented in COCO API.
"""

evaluator = COCOEvaluator("neuron_train", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "neuron_train")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
