import os
import cv2
import torch
import config_test
from utils.loader import CustomDataset
from monai.data import ( DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch,)
from tqdm import tqdm
from model.Network import UIUNET
from utils.misc import overlay, save_mat

test_dataset = CustomDataset(config_test.TEST_FILENAME)
test_ds = CacheDataset(test_dataset, num_workers=0, cache_rate=0.5)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True)
model = UIUNET(3, 1).to(device)
torch.backends.cudnn.benchmark = True
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def clr_2_bw(image, threshold_value):
    (_, blackAndWhiteImage) = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
    return blackAndWhiteImage

model.load_state_dict(torch.load(config_test.BEST_MODEL, map_location=device))
model.eval()
i = 0
with torch.no_grad():
    for batch in epoch_iterator:
        i +=1
        img = batch["image"].type(torch.FloatTensor)
        label = batch["label"].type(torch.FloatTensor)
        d0, d1, d2, d3, d4, d5, d6 = model(img.cuda())
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = clr_2_bw(pred.permute(1, 2, 0).cpu().numpy(), threshold_value=0.8)
        img = img.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()
        label = label.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()

        overlay_prdctd = overlay(img,pred)
        overlay_lbld = overlay(img, label)
        save_mat(file=overlay_prdctd , i = i, dir = config_test.TEST_DIR, folder_name = "overlay_predictions")
        save_mat(file=overlay_lbld, i=i, dir=config_test.TEST_DIR, folder_name="overlay_labels")
        save_mat(file=img, i=i, dir=config_test.TEST_DIR, folder_name="images")
        save_mat(file=label, i=i, dir=config_test.TEST_DIR, folder_name="labels")
        save_mat(file=pred, i=i, dir=config_test.TEST_DIR, folder_name="predictions")
    print("Testing completed")


