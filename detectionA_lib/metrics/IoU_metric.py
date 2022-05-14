import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def IoU(bbox_true, bbox_pred):
    xt_left, xt_right = bbox_true[0], bbox_true[0] + bbox_true[2]
    xp_left, xp_right = bbox_pred[0], bbox_pred[0] + bbox_pred[2]
    yt_low, yt_high = bbox_true[1], bbox_true[1] + bbox_true[3]
    yp_low, yp_high = bbox_pred[1], bbox_pred[1] + bbox_pred[3]
    
    is_intersect = False if xp_right < xt_left or xt_right < xp_left else True
    if yt_high < yp_low or yp_high < yt_low:
        is_intersect = False
    if is_intersect:
        intersect = (min(xt_right, xp_right) - max(xt_left, xp_left))*(min(yt_high, yp_high) - max(yt_low, yp_low))
        union = (xt_right-xt_left)*(yt_high-yt_low)+(xp_right-xp_left)*(yp_high-yp_low) - intersect
        return intersect/union
    else:
        return 0

def measure_meanIoU(test_dataloader, detector):
    meanIoU = 0
    img_qty = 0
    detector.eval()
    for path_batch, pic_batch, label_batch in test_dataloader:
        batch_size = label_batch.shape[0]
        pic_batch = pic_batch.to(device)
        truth_bboxes = label_batch.to(device)[:, 3:].to(device)
        pred_bboxes = detector.predict(pic_batch)[:, 1:].to(device)
        for i in range(batch_size):
            label_bbox = truth_bboxes[i, :]
            pred_bbox = pred_bboxes[i, :]
            meanIoU = (meanIoU*img_qty + IoU(label_bbox, pred_bbox))/(img_qty+1)
            img_qty += 1
    return meanIoU
