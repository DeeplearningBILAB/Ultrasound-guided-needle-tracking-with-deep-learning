from monai.transforms import (Compose, RandFlipd,  RandShiftIntensityd)


TRAIN_FILENAME = "/home/xinglab11/Praveen/PA_US/Dataset_train_13_7/train/train_set.npz"
VALID_FILENAME = "/home/xinglab11/Praveen/PA_US/Dataset_train_13_7/val/valid_set.npz"
MODEL_NAME = "best_metric_model.pth"
MODEL_DIR  = "/home/xinglab11/Praveen/PA_US/model"

DEVICE_IDX = 0
MAX_ITERATIONS = 20000
EVALUATION_NUM = 100

GLOBAL_STEP = 0

TRAIN_TRANSFORMS = Compose([ RandFlipd(keys=["image", "label"],spatial_axis=[0], prob=0.10,),RandFlipd(keys=["image", "label"],spatial_axis=[1], prob=0.10,),RandShiftIntensityd(
    keys=["image"], offsets=0.10, prob=0.50,)])