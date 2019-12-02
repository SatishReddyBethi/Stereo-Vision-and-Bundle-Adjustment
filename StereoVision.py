import pyAprilTag
import cv2
import numpy as np

# region Camera Calibration Matrix
K = np.load("Data/K_external.npy")
# Distortion Parameters
D = np.load("Data/K_distCoeffs.npy")
# endregion

