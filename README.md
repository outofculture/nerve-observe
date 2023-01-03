## Detecting neurons in 2-photon microscope images

This attempts to turn the data from http://neurofinder.codeneuro.org/ into a NN that can detect
neurons in live camera feeds.

### Usage

Setup a separate environment, conda or otherwise.

```bash
pip install -r requirements.txt
```

Download and unzip data from http://neurofinder.codeneuro.org/ into a `data/` directory. Also
put one of them into `data/val/`. Then run:

```bash
python train.py
python validate.py
```

If all goes well, that should show you a bunch of bounding boxes around valid neurons.