import numpy as np
import cv2
import os


TAG_FLOAT = 202021.25


def clean_dst_file(dst_file):
    """Create the output folder, if necessary; empty the output folder of previous predictions, if any
    Args:
        dst_file: Destination path
    """
    # Create the output folder, if necessary
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)
        

def flow_read(src_file):
    """Read optical flow stored in a .flo, .pfm, or .png file
    Args:
        src_file: Path to flow file
    Returns:
        flow: optical flow in [h, w, 2] format
    Refs:
        - Written by Phil Ferriere
        - Interpret bytes as packed binary data
        Per https://docs.python.org/3/library/struct.html#format-characters:
        format: f -> C Type: float, Python type: float, Standard size: 4
        format: d -> C Type: double, Python type: float, Standard size: 8
    Based on:
        - To read optical flow data from 16-bit PNG file:
        https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py
        Written by Clément Pinard, Copyright (c) 2017 Clément Pinard
        MIT License
        - To read optical flow data from PFM file:
        https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
        Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li
        License Unknown
        - To read optical flow data from FLO file:
        https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    # Read in the entire file, if it exists
    assert(os.path.exists(src_file))

    if src_file.lower().endswith('.flo'):

        with open(src_file, 'rb') as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert(tag == TAG_FLOAT)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith('.png'):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = (flow_raw[:, :, 0] == 0)
        flow[invalid, :] = 0

    elif src_file.lower().endswith('.pfm'):

        with open(src_file, 'rb') as f:

            # Parse .pfm file header
            tag = f.readline().rstrip().decode("utf-8")
            assert(tag == 'PF')
            dims = f.readline().rstrip().decode("utf-8")
            w, h = map(int, dims.split(' '))
            scale = float(f.readline().rstrip().decode("utf-8"))

            # Read in flow data and reshape it
            flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
            flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
            flow = np.flipud(flow)
    else:
        raise IOError

    return flow


def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    Note: by Phil Ferriere
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)
        
        
def compute_unit_image_gradient(image):
	"""Computes unit image gradient
	Args:
		image: The image to compute gradient of. RGB image.
	"""
	gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	dx, dy = cv2.spatialGradient(gray_frame)
	img_grad = np.stack([dx, dy], axis=-1)
	img_grad = np.divide(img_grad, np.linalg.norm(img_grad, axis=-1, keepdims=True) + 1e-8, dtype=np.float32)
	
	return img_grad