"""
Train us up a neuron detector! Based on:
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
"""

import detectron2
import torch
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

import cfg
from data_mangling import neuron_metadata


assert "neuron" in neuron_metadata.thing_classes, "Could not load data"  # load bearing assertion

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
setup_logger()

# """# Run a pre-trained detectron2 model
#
# We first download an image from the COCO dataset:
# """
#
# # !wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
# example_im = cv2.imread("./data/neurofinder.00.00/images/image00000.tiff")
# cv2.imshow("here's what we're working with", example_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# """Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image."""
#
# test_cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# test_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# test_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# test_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(test_cfg)
# outputs = predictor(example_im)
#
# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
#
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(example_im[:, :, ::-1], MetadataCatalog.get(test_cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("untrained detection (should not find anything)", out.get_image()[:, :, ::-1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""Register the neuron dataset to detectron2, following the
[detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).
Here, the dataset is in its custom format, therefore we write a function to parse it and prepare it into COCO
standard format. User should write such a function when using a dataset in custom format. 
"""

# if your dataset is already in COCO format, this can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

if __name__ == "__main__":
    """## Train!

    Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the neuron dataset. It takes ~2 minutes to train 300
    iterations on a P100 GPU.
    """
    trainer = DefaultTrainer(cfg.cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Look at training curves in tensorboard?
    # %load_ext tensorboard
    # %tensorboard --logdir output
