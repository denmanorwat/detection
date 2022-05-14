from torch.nn import BCELoss
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def BCEDetectionLoss(predictions_batch, targets_batch):
    detection_loss = 0
    bce_fn = BCELoss(reduction = "mean")
    target_anchor_idx, target_grid_cell_x, target_grid_cell_y = targets_batch[:, 0].squeeze(), targets_batch[:, 1].squeeze(), targets_batch[:, 2].squeeze()
    for i, anchor_params in enumerate(predictions_batch):
        target_logits =\
        anchor_params[target_anchor_idx == i, 0, target_grid_cell_x[target_anchor_idx == i], target_grid_cell_y[target_anchor_idx == i]]
        target_bboxes =\
        anchor_params[target_anchor_idx == i, 1:, target_grid_cell_x[target_anchor_idx == i], target_grid_cell_y[target_anchor_idx == i]]
        map_volume = anchor_params[:, 0, :, :].shape[0]*anchor_params[:, 0, :, :].shape[1]*anchor_params[:, 0, :, :].shape[2]
        detection_loss += bce_fn(anchor_params[:, 0, :, :], torch.zeros(anchor_params[:, 0, :, :].shape).to(device))
        if len(target_logits) == 0:
            continue 
        detection_loss -= bce_fn(target_logits, torch.zeros(target_logits.shape).to(device))/map_volume
        detection_loss += bce_fn(target_logits, torch.ones(target_logits.shape).to(device))
        detection_loss += torch.abs(torch.log(target_bboxes+1e-04) - torch.log(targets_batch[target_anchor_idx == i, 3:]+1e-04).to(device)).mean()
    return detection_loss
