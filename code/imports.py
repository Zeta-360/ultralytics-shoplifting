import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
import sys
import os
import torch
import roboflow
import base64

from roboflow import Roboflow
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import uuid
from einops import rearrange, repeat