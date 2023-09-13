import os
import torch
import config
from tqdm import tqdm
from model.Network import UIUNET
from matplotlib import pyplot as plt
from utils.loader  import CustomDataset
from utils.loss import muti_bce_loss_fusion
from monai.data import (DataLoader, CacheDataset)


train_dataset = CustomDataset(config.TRAIN_FILENAME)
train_ds = CacheDataset(train_dataset, transform=config.TRAIN_TRANSFORMS ,num_workers=4, cache_rate=0.5)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_dataset = CustomDataset(config.VALID_FILENAME)
val_ds = CacheDataset(val_dataset, num_workers=0, cache_rate=0.5)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(f"cuda:{config.DEVICE_IDX}" if torch.cuda.is_available() else "cpu")
model = UIUNET(3, 1).to(device)

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def validation(epoch_iterator_val):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inp = batch["image"].type(torch.FloatTensor)
            val_lbl = batch["label"].type(torch.FloatTensor)
            val_inputs, val_labels = (val_inp.to(device), val_lbl.to(device))
            dv0, dv1, dv2, dv3, dv4, dv5, dv6 = model(val_inputs)
            loss2, loss = muti_bce_loss_fusion(dv0, dv1, dv2, dv3, dv4, dv5, dv6, val_labels)
            total_loss += loss.item()
            num_batches += 1
    average_loss = total_loss / num_batches
    return average_loss


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training ", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x_img = batch["image"].type(torch.FloatTensor)
        y_img = batch["label"].type(torch.FloatTensor)
        x, y = (x_img.to(device), y_img.to(device))
        d0, d1, d2, d3, d4, d5, d6 = model(x)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, config.MAX_ITERATIONS, loss))
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        if (global_step % config.EVALUATION_NUM == 0 and global_step != 0) or global_step == config.MAX_ITERATIONS:
            epoch_iterator_val = tqdm(val_loader, desc="Validation", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val < dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, config.MODEL_NAME))
                print(
                    "Model Was Saved ! Current Best Avg. Loss: {} Current Avg. Loss: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Loss: {} Current Avg. Loss: {}".format(
                        dice_val_best, dice_val
                    )
                )
            if global_step == config.MAX_ITERATIONS:  # Add an additional check for max_iterations
                break
        global_step += 1
    return global_step, dice_val_best, global_step_best





if __name__ == "__main__":

    epoch_loss_values = []
    metric_values = []
    dice_val_best = 10.0
    global_step_best = 0
    while config.GLOBAL_STEP < config.MAX_ITERATIONS:
        global_step, dice_val_best, global_step_best = train(config.GLOBAL_STEP, train_loader, dice_val_best, global_step_best)
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, config.MODEL_NAME)))
    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [config.EVALUATION_NUM * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Loss")
    x = [config.EVALUATION_NUM * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.show()



