import numpy as np
import argparse
import json
import os
from glob import glob

from detectron2.structures import polygons_to_bitmask, BoxMode

# from matplotlib import pyplot as plt
from numpy import array, zeros
import cv2


def polygon_to_mask(polygon: list, shape=(520, 704)):
    """
    polygon: a list of [[x1, y1], [x2, y2],....]
    shape: shape of bitmask
    Return: RLE type of mask
    """
    # add 0.25 can keep the pixels before and after the conversion unchanged
    polygon = [x_or_y for coords in polygon for x_or_y in coords]
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1])
    # rle = mask_util.encode(np.asfortranarray(mask))
    # return rle
    return mask


def bounding_box(points):
    """ return the bounding box of a set of points"""
    return np.array([np.min(points, axis=0), np.max(points, axis=0)])


def main(params):
    os.chdir(params.data_dir)
    # load the images
    files = sorted(glob("images/*.tiff"))
    imgs = array([cv2.imread(f) for f in files])
    dims = imgs.shape[1:]
    assert dims, "could not load images from specified directory (should be e.g. neurofinder.00.00)"

    # load the regions from across the _entire_ time-series
    with open("regions/regions.json") as f:
        regions = json.load(f)  # list of {id: int, coordinates: List[List[int, int]]}.
        # id is arbitrary. coordinates describe the neuron's polygon boundary

    masks_and_bboxes_and_regions = array(
        [(polygon_to_mask(s["coordinates"], dims), bounding_box(s["coordinates"]), s["coordinates"]) for s in regions]
    )

    cutoff = 1.4  # todo: so arbitrary! can we get this value from the image somehow?
    # figure out which regions apply to each image.
    for img in imgs:
        img_annots = [
            {"bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS, "segmentation": region, "category_id": 0, "iscrowd": False}
            for mask, bbox, region in masks_and_bboxes_and_regions
            if np.mean(img[mask]) > cutoff
        ]

    # todo load data
    # todo train transformer
    # todo save model
    # todo test model

    # # show the outputs?
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(imgs.sum(axis=0), cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(masks.sum(axis=0), cmap='gray')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=False, help="data directory", default="data")
    main(parser.parse_args())
