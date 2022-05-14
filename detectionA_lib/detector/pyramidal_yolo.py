from torch.nn.modules.container import ModuleList
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InceptionBlock(torch.nn.Module):
    def __init__(self, *modules):
        super(InceptionBlock, self).__init__()
        module_list = ModuleList()
        for module in modules:
            module_list.append(module)
        self.module_list = module_list
    
    def __call__(self, x):
        y = None
        for module in self.module_list:
            y = module(x) if y is None else y + module(x)
        return y

class ResNetPyramidDetector(torch.nn.Module):
    def __resBlock(layers_in_resblock, starting_fmap_size, ending_fmap_size, prep_layer = False):
        modules = ModuleList()
        long_list = ModuleList()
        long_list.append(nn.Conv2d(starting_fmap_size, ending_fmap_size, kernel_size = 3, padding = 1))
        long_list.append(nn.BatchNorm2d(ending_fmap_size))
        long_list.append(nn.ReLU())
        for layer in range(layers_in_resblock-1):
            long_list.append(nn.Conv2d(ending_fmap_size, ending_fmap_size, kernel_size = 3, padding = 1))
            long_list.append(nn.BatchNorm2d(ending_fmap_size))
            long_list.append(nn.ReLU())
            if layer % 2 == 0:
                shortcut = torch.nn.Identity() if layer != 0 else torch.nn.Conv2d(starting_fmap_size, ending_fmap_size, kernel_size = 1)
                long_list = torch.nn.Sequential(*long_list)
                modules.append(InceptionBlock(long_list, shortcut))
                long_list = []
        if not prep_layer:
            modules.append(nn.Conv2d(ending_fmap_size, ending_fmap_size, kernel_size = 2, stride = 2))
        resBlock = torch.nn.Sequential(*modules)
        return resBlock
    
    def __detectorBlock(preparation_layers = 2):
        modules = ModuleList()
        modules.append(ResNetPyramidDetector.__resBlock(preparation_layers, 256, 256, prep_layer = True))
        # 4 points for box regression and 1 point for probability of a face. 
        # Quantity of anchors used: 1
        modules.append(nn.Conv2d(256, 5, kernel_size = 1))
        detectorBlock = torch.nn.Sequential(*modules)
        return detectorBlock

    def __get_best_bbox(prediction_tensors):
    	batch_size = prediction_tensors[0].shape[0]
    	glob_best_tensor_prob = torch.zeros(batch_size).to(device).to(torch.float32)
    	glob_best_tensor_idx = torch.zeros(batch_size).to(device).to(torch.long)
    	glob_best_tensor_x_coord = torch.zeros(batch_size).to(device).to(torch.long)
    	glob_best_tensor_y_coord = torch.zeros(batch_size).to(device).to(torch.long)
    	for i in range(len(prediction_tensors)):
        	prediction_tensor = prediction_tensors[i]
        	width = prediction_tensor.shape[3]

        	loc_best_tensor_prob, loc_best_tensor_idx_flattened = prediction_tensor[:, 0, :, :].view(batch_size, -1).max(dim = 1)
        	loc_best_tensor_x_coord = loc_best_tensor_idx_flattened%width
        	loc_best_tensor_y_coord = loc_best_tensor_idx_flattened//width

        	glob_best_tensor_idx[loc_best_tensor_prob > glob_best_tensor_prob] = i
        	glob_best_tensor_x_coord[loc_best_tensor_prob > glob_best_tensor_prob] = loc_best_tensor_x_coord[loc_best_tensor_prob > glob_best_tensor_prob]
        	glob_best_tensor_y_coord[loc_best_tensor_prob > glob_best_tensor_prob] = loc_best_tensor_y_coord[loc_best_tensor_prob > glob_best_tensor_prob]
        	glob_best_tensor_prob[loc_best_tensor_prob > glob_best_tensor_prob] = loc_best_tensor_prob[loc_best_tensor_prob > glob_best_tensor_prob]
    	batch_bbox_pred = []
    	for idx_of_pic, tensor_idx_x_y in enumerate(zip(glob_best_tensor_idx, glob_best_tensor_x_coord, glob_best_tensor_y_coord)):
        	tensor_idx, x_coord, y_coord = tensor_idx_x_y
        	prob = prediction_tensors[tensor_idx][idx_of_pic, 0, y_coord, x_coord]
        	x1, y1, w, h = prediction_tensors[tensor_idx][idx_of_pic, 1:, y_coord, x_coord]
        	batch_bbox_pred.append(torch.tensor([prob.item(), x1.item(), y1.item(), w.item(), h.item()]))
    	return torch.stack(batch_bbox_pred)
        
    def __init__(self, starting_fmap_qty = 64, ending_fmap_qty = 256, layers_in_prep_resblock = 2, layers_in_resize_resblock = 4):
        super(ResNetPyramidDetector, self).__init__()
        self.preparation_layers = ModuleList()
        self.preparation_layers.append(nn.Sequential(nn.Conv2d(kernel_size = 7, stride = 2, in_channels = 3, out_channels = starting_fmap_qty, padding = 3), 
                                                     nn.MaxPool2d(kernel_size = 2, stride = 2)))
        # Receptive field: 3 -> 64 -> 128 -> 256: 3 transitions
        # After each transition reception field enlargens by layers_in_prep_resblock
        # Receptive field after preparation layer: 3 * layer_in_prep_resblock*(kernel_size-1): 3*4*2=24
        # Thus, receptive fields of detectors are: (rec_field_after_prep_layer+layers_in_resize_block*(kernel_size-1))*stride = (24+6*2)*2 = 72,
        # (72 + 6*2)*2 = 168, (168+6*2)*2 = 360, (360+6*2)*2 = 744
        while(starting_fmap_qty < ending_fmap_qty):
            self.preparation_layers.append(ResNetPyramidDetector.__resBlock(layers_in_prep_resblock, starting_fmap_qty, starting_fmap_qty*2, prep_layer = True))
            starting_fmap_qty = 2*starting_fmap_qty if 2*starting_fmap_qty <= ending_fmap_qty else ending_fmap_qty
        self.preparation_layers = torch.nn.Sequential(*self.preparation_layers)
        self.resizer_layer_2 = ResNetPyramidDetector.__resBlock(layers_in_resize_resblock, ending_fmap_qty, ending_fmap_qty, prep_layer = False)
        self.resizer_layer_4 = ResNetPyramidDetector.__resBlock(layers_in_resize_resblock, ending_fmap_qty, ending_fmap_qty, prep_layer = False)
        self.resizer_layer_8 = ResNetPyramidDetector.__resBlock(layers_in_resize_resblock, ending_fmap_qty, ending_fmap_qty, prep_layer = False)
        self.resizer_layer_16 = ResNetPyramidDetector.__resBlock(layers_in_resize_resblock, ending_fmap_qty, ending_fmap_qty, prep_layer = False)
        self.small_detector = ResNetPyramidDetector.__detectorBlock()
        self.not_so_small_detector = ResNetPyramidDetector.__detectorBlock()
        self.not_so_big_detector = ResNetPyramidDetector.__detectorBlock()
        self.big_detector = ResNetPyramidDetector.__detectorBlock()

    def __call__(self, x):
        x = self.preparation_layers(x)
        x_2 = self.resizer_layer_2(x)
        x_4 = self.resizer_layer_4(x_2)
        x_8 = self.resizer_layer_8(x_4)
        x_16 = self.resizer_layer_16(x_8)

        rich_x_16 = x_16
        rich_x_8 = x_8 + nn.Upsample(size=(x_8.shape[-2:]))(x_16)
        rich_x_4 = x_4 + nn.Upsample(size=(x_4.shape[-2:]))(x_8)
        rich_x_2 = x_2 + nn.Upsample(size=(x_2.shape[-2:]))(x_4)

        big_objs = self.big_detector(rich_x_16)
        big_objs[:, 0, :, :] = torch.sigmoid(big_objs[:, 0, :, :])
        big_objs[:, 1:, :, :] = torch.exp(big_objs[:, 1:, :, :])
        not_so_big_objs = self.not_so_big_detector(rich_x_8)
        not_so_big_objs[:, 0, :, :] = torch.sigmoid(not_so_big_objs[:, 0, :, :])
        not_so_big_objs[:, 1:, :, :] = torch.exp(not_so_big_objs[:, 1:, :, :])
        not_so_small_objs = self.not_so_small_detector(rich_x_4)
        not_so_small_objs[:, 0, :, :] = torch.sigmoid(not_so_small_objs[:, 0, :, :])
        not_so_small_objs[:, 1:, :, :] = torch.exp(not_so_small_objs[:, 1:, :, :])
        small_objs = self.small_detector(rich_x_2)
        small_objs[:, 0, :, :] = torch.sigmoid(small_objs[:, 0, :, :])
        small_objs[:, 1:, :, :] = torch.exp(small_objs[:, 1:, :, :])
        
        return [small_objs, not_so_small_objs, not_so_big_objs, big_objs]
    
    def predict(self, x):
        pic_h, pic_w = x.shape[2], x.shape[3]
        x = self.preparation_layers(x)
        x_2 = self.resizer_layer_2(x)
        x_4 = self.resizer_layer_4(x_2)
        x_8 = self.resizer_layer_8(x_4)
        x_16 = self.resizer_layer_16(x_8)

        rich_x_16 = x_16
        rich_x_8 = x_8 + nn.Upsample(size=(x_8.shape[-2:]))(x_16)
        rich_x_4 = x_4 + nn.Upsample(size=(x_4.shape[-2:]))(x_8)
        rich_x_2 = x_2 + nn.Upsample(size=(x_2.shape[-2:]))(x_4)

        big_objs = self.big_detector(rich_x_16)
        big_objs[:, 0, :, :] = torch.sigmoid(big_objs[:, 0, :, :])
        big_objs[:, 1:, :, :] = torch.exp(big_objs[:, 1:, :, :])
        not_so_big_objs = self.not_so_big_detector(rich_x_8)
        not_so_big_objs[:, 0, :, :] = torch.sigmoid(not_so_big_objs[:, 0, :, :])
        not_so_big_objs[:, 1:, :, :] = torch.exp(not_so_big_objs[:, 1:, :, :])
        not_so_small_objs = self.not_so_small_detector(rich_x_4)
        not_so_small_objs[:, 0, :, :] = torch.sigmoid(not_so_small_objs[:, 0, :, :])
        not_so_small_objs[:, 1:, :, :] = torch.exp(not_so_small_objs[:, 1:, :, :])
        small_objs = self.small_detector(rich_x_2)
        small_objs[:, 0, :, :] = torch.sigmoid(small_objs[:, 0, :, :])
        small_objs[:, 1:, :, :] = torch.exp(small_objs[:, 1:, :, :])

        pred_list = [small_objs, not_so_small_objs, not_so_big_objs, big_objs]
        best_bboxes = ResNetPyramidDetector.__get_best_bbox(pred_list)
        return best_bboxes
