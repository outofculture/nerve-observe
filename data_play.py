import json
import os
import random
from functools import cache, reduce
from glob import glob
from typing import Optional, Dict, List

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import SpinBox, PolyLineROI
from pyqtgraph.Qt import QtWidgets

from data_mangling import region_info, get_local_value


@cache
def get_region_info(plate_dir):
    img = random.choice(glob(os.path.join(plate_dir, "images", "*.tiff")))
    dims = cv2.imread(img).shape
    with open(os.path.join(plate_dir, "regions", "regions.json")) as f:
        return [region_info(s["coordinates"], dims) for s in json.load(f)]


def show_regions_changed(should_show):
    global img_view, img_data, img_data_orig
    # TODO do this without changing the pos/zoom
    if should_show:
        img_view.setImage(img_data)
    else:
        img_view.setImage(img_data_orig)
    img_view.updateImage()


def update_region_display():
    global cutoff, img_data, img_view, img_data_orig, regions, rois, non_neuronal_mask

    if img_data is not None:
        img_data[:] = img_data_orig.copy()
        for roi in rois:
            img_view.removeItem(roi)
        rois.clear()
        for info in regions:
            local_value = get_local_value(img_data, info, non_neuronal_mask)
            if np.mean(img_data[info["mask"]]) - local_value > cutoff:
                img_data[info["mask"]] = [10, 1, 5]
                # TODO this doesn't work
                # poly = zip(info["polygon"][1::2], info["polygon"][::2])
                # roi = PolyLineROI(positions=poly, closed=True)
                # roi.show()
                # img_view.addItem(roi)
                # rois.append(roi)
        show_regions.setChecked(True)
        show_regions_changed(True)


def change_cutoff(new_val):
    global cutoff
    cutoff = new_val
    update_region_display()


def load_new_image():
    global regions, cutoff, img_data, img_data_orig, img_view, filename_label, show_regions, non_neuronal_mask
    plate_dir = random.choice(glob(os.path.join("data", "neurofinder*")))
    # img = random.choice(glob(os.path.join(plate_dir, "images", "*.tiff")))
    # filename_label.setText(f"{plate_dir}/{img}")
    filename_label.setText(f"{plate_dir}")
    img_data = np.max(np.array([cv2.imread(i) for i in glob(os.path.join(plate_dir, "images", "*.tiff"))]), axis=0)
    # img = random.choice(glob(os.path.join(plate_dir, "images", "*.tiff")))
    # img_data = cv2.imread(img)
    img_data_orig = img_data.copy()
    img_view.setImage(img_data)
    regions = get_region_info(plate_dir)
    non_neuronal_mask = np.logical_not(
        reduce(lambda mask, info: mask | info["mask"], regions, np.zeros(img_data.shape[:2]).astype(bool))
    )
    cutoff = 0.9
    cutoff_spinner.setValue(cutoff)
    show_regions.setChecked(True)
    update_region_display()


pg.setConfigOptions(imageAxisOrder="row-major")
app = pg.mkQApp("Play with neurofinder data annotations")
win = QtWidgets.QMainWindow()
win.resize(800, 800)
win.setWindowTitle("Play with neurofinder data annotations")

centralwidget = QtWidgets.QWidget(win)
win.setCentralWidget(centralwidget)  # todo superfluous?
centralwidget.setObjectName("centralwidget")
ui_layout = QtWidgets.QGridLayout(centralwidget)

img_view = pg.ImageView()
ui_layout.addWidget(img_view, 0, 0, 3, 2)

new_img_button = QtWidgets.QPushButton()
ui_layout.addWidget(new_img_button, 3, 0)
new_img_button.setText("Load new image")
new_img_button.clicked.connect(load_new_image)

filename_label = QtWidgets.QLabel("")
ui_layout.addWidget(filename_label, 3, 1)

regions: List[Dict] = []
img_data: Optional[np.ndarray] = None
img_data_orig: Optional[np.ndarray] = None
cutoff: float = 0
rois: List[PolyLineROI] = []
non_neuronal_mask: Optional[np.ndarray] = None

show_regions = QtWidgets.QCheckBox("Show regions")
show_regions.setChecked(True)
show_regions.stateChanged.connect(show_regions_changed)
ui_layout.addWidget(show_regions, 4, 0)


cutoff_spinner = SpinBox()
cutoff_spinner.setMinimum(0)
cutoff_spinner.setMaximum(255)
cutoff_spinner.setValue(0)
ui_layout.addWidget(cutoff_spinner, 4, 1)
cutoff_spinner.valueChanged.connect(change_cutoff)

# todo display the np.mean of the image
# todo hook a spinner up to the cutoff for which regions to include


if __name__ == "__main__":
    load_new_image()
    win.show()
    pg.exec()
