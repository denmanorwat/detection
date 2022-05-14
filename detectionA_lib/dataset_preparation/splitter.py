import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_val_test_split(root_folder, loc_path_to_bboxes, val_frac = 0.15, test_frac = 0.15):
    wild_imgs_qty = len(os.listdir(root_folder)) - 1
    train_imgs_qty = int((1-val_frac-test_frac)*wild_imgs_qty)
    val_imgs_qty = int(val_frac*wild_imgs_qty)
    test_imgs_qty = wild_imgs_qty - train_imgs_qty - val_imgs_qty

    work_folder = "celebrities_splitted"
    subfolder_names = ["train", "val", "test"]
    subfolder_capacities = [train_imgs_qty, val_imgs_qty, test_imgs_qty]
    with open(root_folder + "/" + loc_path_to_bboxes, "r") as bbox_file:
            # Info about qty of imgs
            cur_line = bbox_file.readline()
            # Header of a table
            cur_line = bbox_file.readline()
            for subfolder_name, subfolder_capacity in zip(subfolder_names, subfolder_capacities):
                os.makedirs(work_folder + "/" + subfolder_name)
                with open(work_folder + "/" + subfolder_name + "/" + "bbox.txt", "w") as subfolder_bbox_file:
                    for i in range(subfolder_capacity): # 
                        line = bbox_file.readline().split()
                        img_header, x1, y1, w, h = line[0], line[1], line[2], line[3], line[4]
                        subfolder_bbox_file.write(img_header + " " + x1 + " " + y1 + " " + w + " " + h + "\n")
                        os.rename(root_folder + "/" + img_header, work_folder + "/" + subfolder_name + "/" + img_header)
