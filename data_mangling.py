import json
import os
import random
import re
from functools import reduce
from glob import glob
from typing import List, Dict

import cv2
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from pycocotools import mask as mask_utils
from tqdm import tqdm

from shaunna_is_the_best import alt_poly_to_mask


def polygon_to_mask(polygon: List[List], shape):
    """
    polygon: a list of [[x₁, y₁], [x₂, y₂],....]
    shape: shape of bitmask
    Return: shape-shaped boolean array with True in all in-poly points
    source: https://www.kaggle.com/code/linrds/convert-rle-to-polygons
    """
    return alt_poly_to_mask(polygon, (shape[0], shape[1]))
    # polygon = [x_or_y + 0.5 for coords in polygon for x_or_y in reversed(coords)]
    # return polygons_to_bitmask([np.asarray(polygon)], shape[0], shape[1])


def polygon_to_rle(polygon: List, shape):
    m = polygon_to_mask(polygon, shape)
    return mask_utils.encode(np.asfortranarray(m))


def bounding_box(points):
    """ return the bounding box of a set of points"""
    return np.array([*np.min(points, axis=0), *np.max(points, axis=0)])


def region_info(region: List, dims: List):
    assert len(region) > 2, f"coordinate list for a polygon must have at least 3 points: {region}"
    return {
        "mask": polygon_to_mask(region, dims),
        "bbox": bounding_box(region),
        "rle": polygon_to_rle(region, dims),
        "polygon": [[x_or_y for coords in region for x_or_y in reversed(coords)]],
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_coco_annotations(plate_dir: str) -> List[Dict]:
    dataset_dicts = []
    plate_num = int(re.sub(r"[^0-9]+", "", plate_dir))
    img_files = glob(os.path.join(plate_dir, "images", "*.tiff"))
    # img_files = random.sample(glob(os.path.join(plate_dir, "images", "*.tiff")), 20)
    dims = cv2.imread(img_files[0]).shape
    assert dims, f"could not load images from {plate_dir}/images/*tiff"

    # the regions are from across the _entire_ time-series
    with open(os.path.join(plate_dir, "regions", "regions.json")) as f:
        # list of {id: int, coordinates: List[List[int, int]]}.
        # id is arbitrary. coordinates are backwards: y, x.
        regions = [region_info(s["coordinates"], dims) for s in json.load(f)]

    with tqdm(total=len(img_files)) as pbar:
        for img in img_files:
            img_num = re.sub(r"[^0-9]+", "", img)
            img_num = plate_num * (10 ** len(img_num)) + int(img_num)  # they're nicely zero-padded
            img_data = cv2.imread(img)
            non_neuronal_mask = np.logical_not(
                reduce(lambda mask, i: mask | i["mask"], regions, np.zeros(img_data.shape[:2]).astype(bool))
            )
            cutoff = 1.3
            # todo ooo, what if we looked at the _local_ neighborhood instead of the entire image.
            # todo maybe even masking out other possible neurons first...
            # todo: so arbitrary!
            # todo: account for bleaching?
            # todo: maybe look at the max of the entire time series?
            annotations = []
            for info in regions:
                local_value = get_local_value(img_data, info, non_neuronal_mask)
                if np.mean(img_data[info["mask"]]) - local_value > cutoff:
                    annotations.append({
                        "image_id": img_num,
                        "bbox": info["bbox"],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": info["polygon"],
                        "category_id": 0,
                        "iscrowd": False,
                    })

            dataset_dicts.append(
                {"id": img_num, "file_name": img, "height": dims[0], "width": dims[1], "annotations": annotations}
            )
            pbar.update(1)
        return dataset_dicts


def get_local_value(img_data, info, non_neuronal_mask):
    local_bbox = (info["bbox"] + [-20, -20, 20, 20]).clip(0)
    local_mask = non_neuronal_mask[local_bbox[0]: local_bbox[2], local_bbox[1]: local_bbox[3]]
    local_value = np.mean(img_data[local_bbox[0]: local_bbox[2], local_bbox[1]: local_bbox[3]][local_mask])
    return local_value


def get_neuron_dicts(data_dir):  # contains e.g. neurofinder.00.00/
    annotations = []
    for plate_dir in glob(os.path.join(data_dir, "neurofinder*")):
        output_file = os.path.join(plate_dir, "generated_coco_annotations.json")
        if os.path.exists(output_file) and os.path.getmtime(output_file) > os.path.getmtime(__file__):
            with open(output_file) as f:
                annotations.extend(json.load(f))
        else:
            plate_annots = make_coco_annotations(plate_dir)
            with open(output_file, 'w') as f:
                json.dump(plate_annots, f, cls=NumpyEncoder)
            annotations.extend(plate_annots)

    return annotations


training_dicts = get_neuron_dicts("data")
# DatasetCatalog.register("neuron_train", lambda x="train": random.sample(training_dicts, 2000))
DatasetCatalog.register("neuron_train", lambda x="train": training_dicts)
MetadataCatalog.get("neuron_train").set(thing_classes=["neuron"])
# DatasetCatalog.register("neuron_val", lambda x="val": get_neuron_dicts("data/val"))
# MetadataCatalog.get("neuron_val").set(thing_classes=["neuron"])

neuron_metadata = MetadataCatalog.get("neuron_train")

if __name__ == "__main__":
    """To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the
    training set:
    """
    for d in random.sample(training_dicts, 2):
        example_im = cv2.imread(d["file_name"])
        visualizer = Visualizer(example_im[:, :, ::-1], metadata=neuron_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("training data", example_im * 2 / np.max(example_im))
        cv2.imshow("annotations (do you see corresponding smudges?)", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
