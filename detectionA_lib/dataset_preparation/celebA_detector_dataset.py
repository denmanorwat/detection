import torch
from torchvision import transforms
from PIL import Image
from math import log2
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class wildCelebDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_imgs, path_to_bboxes, min_square_side):
        self.path_to_imgs = path_to_imgs
        self.min_square_side = min_square_side
        self.target_tensor_params = wildCelebDataset.prepare_targets(path_to_imgs, path_to_bboxes, self.min_square_side)
        self.image_headers = list(self.target_tensor_params.keys())
        self.to_tensor = transforms.ToTensor()

    def __preparePaddingParamsAndScale(img_width, img_height, bbox_width, bbox_height, min_square_side):
        longest_img_side = max(img_width, img_height)
        square_side = max(min_square_side, longest_img_side)
        delta_left = (square_side - img_width)//2
        delta_right = square_side - img_width - delta_left
        delta_higher = (square_side - img_height)//2
        delta_lower = square_side - img_height - delta_higher
        face_percentage = bbox_width*bbox_height/(longest_img_side**2)
        scale = max(longest_img_side/min_square_side, 1)
        return (delta_left, delta_right, delta_higher, delta_lower, face_percentage, scale)

        
    def prepare_targets(path_to_imgs, path_to_bboxes, square_side, face_frac_threshold = 0.08):
        with open(path_to_bboxes, "r") as bbox_file:
            # Current image
            cur_line = bbox_file.readline()
            to_tensor = transforms.ToTensor()
            img_to_params = {}
            while cur_line != '':
                # Data about bbox and image
                img_header, x_1, y_1, bbox_width, bbox_height = cur_line.split(" ")
                x_1, y_1, bbox_width, bbox_height = int(x_1), int(y_1), int(bbox_width), int(bbox_height)
                with Image.open(path_to_imgs + "/" + img_header) as img:
                    img_tensor = to_tensor(img)
                img_height, img_width = img_tensor.shape[1], img_tensor.shape[2]
                delta_left, delta_right, delta_higher, delta_lower, face_percentage, scale =\
                wildCelebDataset.__preparePaddingParamsAndScale(img_width, img_height, bbox_width, bbox_height, square_side)
                x_1, y_1, bbox_width, bbox_height = int((x_1 + delta_left)/scale), int((y_1+delta_lower)/scale), int(bbox_width/scale), int(bbox_height/scale)

                x_1_center, y_1_center = x_1+bbox_width//2, y_1+bbox_height//2
                # Selection of target anchor
                anchor_type = max(int(log2(face_percentage/(face_frac_threshold/2)+0.001)), 0) if face_percentage < face_frac_threshold*2**3 else 3
                grid_size = 2**(anchor_type + 1 + 2)
                target_anchor_idx = anchor_type
                # Selection of target tensor params
                target_grid_cell_x, target_grid_cell_y = x_1_center//grid_size, y_1_center//grid_size
                img_to_params[img_header] = (target_anchor_idx, target_grid_cell_x, target_grid_cell_y,
                                             x_1, y_1, bbox_width, bbox_height, 
                                             delta_left, delta_right, delta_lower, delta_higher)
                cur_line = bbox_file.readline()
        return img_to_params
    
    def __len__(self):
        return len(self.target_tensor_params)

    def __getitem__(self, idx):
        img_header = self.image_headers[idx]
        target_anchor, target_grid_cell_x, target_grid_cell_y,\
        bbox_x_1, bbox_y_1, bbox_width, bbox_height,\
        delta_left, delta_right, delta_lower, delta_higher = self.target_tensor_params[img_header]
        add_padding = transforms.Pad(padding = (delta_left, delta_higher, delta_right, delta_lower))
        resize = transforms.Resize(self.min_square_side)
        with Image.open(self.path_to_imgs + "/" + img_header) as img:
            pic_tensor = resize(add_padding(self.to_tensor(img)))
        pic_width, pic_height = pic_tensor.shape[1], pic_tensor.shape[2]
        # 5 elements in anchor; size of an detector output for certain anchor defined by size of a grid.
        target_coordinates_tensor = torch.tensor([target_anchor, target_grid_cell_x, target_grid_cell_y, bbox_x_1, bbox_y_1, bbox_width, bbox_height])
        return (img_header, pic_tensor, target_coordinates_tensor)
