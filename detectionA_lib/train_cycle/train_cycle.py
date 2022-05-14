from google.colab import output
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image
from matplotlib import patches
from torchvision import transforms
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("./detectionA_lib/metrics/IoU_metric.py") as infile:
    exec(infile.read())

def images_and_bboxes(path_to_train, path_to_val, min_square_side):
    train_imgs, train_bboxes = [], []
    with open(path_to_train + "/" + "bbox.txt", "r") as bbox_file:
        for i in range(12):
            img_header, x1, y1, w, h = bbox_file.readline().split()
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            with Image.open(path_to_train + "/" + img_header) as img:
                tensor_img = transforms.ToTensor()(img)
                img_width, img_height = tensor_img.shape[2], tensor_img.shape[1]

                longest_img_side = max(img_width, img_height)
                square_side = max(min_square_side, longest_img_side)
                delta_left = (square_side - img_width)//2
                delta_right = square_side - img_width - delta_left
                delta_higher = (square_side - img_height)//2
                delta_lower = square_side - img_height - delta_higher
                scale = max(longest_img_side/min_square_side, 1)
                x1, y1, w, h = int((x1 + delta_left)/scale), int((y1+delta_lower)/scale), int(w/scale), int(h/scale)

                padder = transforms.Pad(padding = (delta_left, delta_higher, delta_right, delta_lower))
                resizer = transforms.Resize(min_square_side)
                train_imgs.append(resizer(padder(img)))
            train_bboxes.append((x1, y1, w, h))

    val_imgs, val_bboxes = [], []
    with open(path_to_val + "/" + "bbox.txt", "r") as bbox_file:
        for i in range(12):
            img_header, x1, y1, w, h = bbox_file.readline().split()
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            with Image.open(path_to_val + "/" + img_header) as img:
                tensor_img = transforms.ToTensor()(img)
                img_width, img_height = tensor_img.shape[2], tensor_img.shape[1]

                longest_img_side = max(img_width, img_height)
                square_side = max(min_square_side, longest_img_side)
                delta_left = (square_side - img_width)//2
                delta_right = square_side - img_width - delta_left
                delta_higher = (square_side - img_height)//2
                delta_lower = square_side - img_height - delta_higher
                scale = max(longest_img_side/min_square_side, 1)
                x1, y1, w, h = int((x1 + delta_left)/scale), int((y1+delta_lower)/scale), int(w/scale), int(h/scale)

                padder = transforms.Pad(padding = (delta_left, delta_higher, delta_right, delta_lower))
                resizer = transforms.Resize(min_square_side)
                val_imgs.append(resizer(padder(img)))
            val_bboxes.append((x1, y1, w, h))
    return train_imgs, train_bboxes, val_imgs, val_bboxes


def displayProgress(model, path_batch, pred_batch, label_batch, train_history, val_history, epoch, min_square_side,
                    folder_name = None, file_name = None):
    output.clear()
    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(1, epoch + 1), train_history, label = "Train")
    plt.plot(np.arange(1, epoch + 1), val_history, label = "Val")
    plt.savefig("./drive/MyDrive/" + folder_name + "/" + file_name + "_" + "graphs" + "_" + "epoch_=_" + str(epoch))
    plt.legend()
    plt.show()
    train_images, train_bboxes_gr_truth, val_images, val_bboxes_gr_truth = images_and_bboxes("celebrities_splitted" + "/" + "train", 
                                                                           "celebrities_splitted" + "/" + "val", min_square_side)
    model.eval()
    train_batch = torch.stack([transforms.ToTensor()(train_image) for train_image in train_images])
    val_batch = torch.stack([transforms.ToTensor()(val_image) for val_image in val_images])
    train_pred = model.predict(train_batch.to(device))
    val_pred = model.predict(val_batch.to(device))
    train_bbox_pred, train_prob_pred = train_pred[:, 1:], train_pred[:, 0]
    val_bbox_pred, val_prob_pred = val_pred[:, 1:], val_pred[:, 0]

    nrows = 4
    ncols = 6
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(15, 12)

    for i in range(len(ax)):
        images = train_images if i < 2 else val_images
        bbox_gr_truth = train_bboxes_gr_truth if i < 2 else val_bboxes_gr_truth
        bbox_pred = train_bbox_pred if i < 2 else val_bbox_pred
        bbox_prob = train_prob_pred if i < 2 else val_prob_pred
        for j in range(len(ax[i])):
            onedim_idx = (i*len(ax[i])+j)%12
            ax[i][j].imshow(images[onedim_idx])
            gr_x1, gr_y1, gr_w, gr_h = bbox_gr_truth[onedim_idx]
            pred_x1, pred_y1, pred_w, pred_h = bbox_pred[onedim_idx]
            gr_rect = patches.Rectangle((gr_x1, gr_y1), gr_w, gr_h, linewidth=1, edgecolor='g', facecolor='none')
            pred_rect = patches.Rectangle((pred_x1, pred_y1), pred_w, pred_h, linewidth=1, edgecolor='r', facecolor='none')
            ax[i][j].add_patch(gr_rect)
            ax[i][j].add_patch(pred_rect)
            ax[i][j].axis("off")
    plt.savefig("./drive/MyDrive/" + folder_name + "/" + file_name + "_" + "examples" + "_" + "epoch_=_" + str(epoch))
    plt.show()

def save_model_weights(model, epoch, folder_name = None, file_name = None):
    with open("./drive/MyDrive/" + folder_name + "/" + file_name + "_" + "epoch_=_" + str(epoch), "xb") as file:
        pickle.dump(model, file)

def save_current_val_accuracy(model, epoch, val_loader, folder_name, file_name):
    with open("./drive/MyDrive/" + folder_name + "/" + file_name, "a") as file:
        file.write("Mean IoU on epoch " + str(epoch) + " is " + str(measure_meanIoU(val_loader, model)))
    
def train_detector(train_loader, val_loader, epochs, detector, optimizer, scheduler, loss_fn, min_square_side, 
                folder_name = None, loss_name = None, weights_name = None, metric_name = None):
    os.makedirs("./drive/MyDrive/" + folder_name, exist_ok = False)
    train_history, val_history = [], []
    for epoch in range(1, epochs + 1):
        for mode in ["train", "val"]:
            if mode == "train":
                detector.train()
                train_loss = 0
                for _, pic_batch, label_batch in tqdm(train_loader):
                    pic_batch = pic_batch.to(device)
                    label_batch = label_batch.to(device)
                    pic_pred = detector(pic_batch)
                    optimizer.zero_grad()
                    loss = loss_fn(pic_pred, label_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    del pic_batch
                train_history.append(train_loss/len(train_loader))
                save_model_weights(detector, epoch, folder_name, weights_name)

            if mode == "val":
                detector.eval()
                val_loss = 0
                for path_batch, pic_batch, label_batch in val_loader:
                    pic_batch = pic_batch.to(device)
                    label_batch = label_batch.to(device)

                    pic_pred = detector(pic_batch)

                    loss = loss_fn(pic_pred, label_batch)
                    val_loss += loss.item()
                val_history.append(val_loss/len(val_loader))
                save_current_val_accuracy(detector, epoch, val_loader, folder_name, metric_name)
                displayProgress(detector, path_batch, pic_pred, label_batch, train_history, val_history, epoch, min_square_side,
                                folder_name, loss_name)
            torch.cuda.empty_cache()
            gc.collect()
    optimizer.zero_grad()
