#
# Brendan Martin
# unpatch.py

import os, sys
#import numpy as np
#from PIL import Image
from ultralytics import YOLO
import csv
from PIL import Image, ImageDraw
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import shuffle, randint
import torch
import torchvision
import time
import argparse
import math
#from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint

# um/pixel length
UM_PER_PIXEL = 0.7784
UM_PER_PATCH = UM_PER_PIXEL * 832
#0.5945 µm2/pixel
MAX_DET = 30000
Image.MAX_IMAGE_PIXELS = 100000000

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
#    if not rotated:
#        if in_place:
#            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
#        else:
#            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy
    if not rotated:
        if in_place:
            prediction[..., :4] = prediction[..., :4]  # xywh to xyxy
        else:
            prediction = torch.cat((prediction[..., :4], prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
#    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        
        # Use the normalized bounding box area in place of the objectness score
        scores = (x[:,2]-x[:,0]) * (x[:,3]-x[:,1])
        scores = scores - scores.min()
        scores = scores / scores.max()
        
#        if rotated:
#            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
#            i = nms_rotated(boxes, scores, iou_thres)
#        else:
        
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            
        i = i[:max_det]  # limit detections

#        output[xi] = x[i]
        output.append( i )
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

#* (2/(num_images[0]+1))
#* (2/(num_images[1]+1))
def scale_boxes(boxes, num_images, img_ind, img_scale):
    boxes[:,(0,2)] = boxes[:,(0,2)] + (img_scale[0]/2)*img_ind[0] # x
    boxes[:,(1,3)] = boxes[:,(1,3)] + (img_scale[1]/2)*img_ind[1] # y
    return boxes
    
def scale_masks(masks, num_images, img_ind, img_scale):
    for m in range(len(masks)):
#        print(masks[m].shape)
        masks[m] = masks[m] + (img_scale/2)*img_ind # x
#        masks[m] = masks[m] + (img_scale[1]/2)*img_ind[1] # y
    return masks
    
def inference(model, idx, img, size, out_dir):
    # divide size of image by size of patch/2
    num_patches = ( np.array(img.size) / (size/2) ).astype(int)
    
    # Run inference on each image
    box_results = []
    mask_results = []
#    for y0 in range(0, img.size[1], size//2)[:2]:
#        for x0 in range(0, img.size[0], size//2)[:2]:
    for y0 in range(0, img.size[1], size//2)[:4]:
        for x0 in range(0, img.size[0], size//2)[:4]:
            x1, y1 = x0+size, y0+size
#            if x1 >= img.size[0]:
#                x0, x1 = img.size[0]-size, img.size[0]-1
#            if y1 >= img.size[1]:
#                y0, y1 = img.size[1]-size, img.size[1]-1
            
            # save crops
            img_crop = img.crop((x0,y0,x1,y1))
            yc=math.ceil(y0/(size//2))
            xc=math.ceil(x0/(size//2))
#            img_crop.save("{f}/img_{id}_{yc}_{xc}.png".format(f=out_dir, id=idx, yc=yc, xc=xc))
            results = model( img_crop )
            img_ind = np.array((xc,yc))
            
            print(img_crop)
            img_crop.save( "{f}/test.png".format(f=out_dir) )
#            identifier = f.split("_")
            
            # Scale the predictions back to their proper size
            for r in range(len(results)):
                boxes = results[r].boxes.data.clone()
#                print(results[r].masks.xy)
#                img_ind = ( int(identifier[3].split(".")[0]), int(identifier[2]) )
                if boxes.numel() != 0: # if osteoclasts detected
                    boxes = scale_boxes(boxes, num_patches, img_ind, (size,size))
                    masks = scale_masks(results[r].masks.xy, num_patches, img_ind, np.array((size,size)))
                    box_results.append( boxes )
                    mask_results += masks
    
    objects_found = True if box_results else False
    
    if objects_found:
        box_results = torch.cat(box_results)
    
    #    print(box_results[:,4])
    #    box_results[:,4] = (box_results[:,2]-box_results[:,0]) * (box_results[:,3]-box_results[:,1])
    #    box_results[:,4] = box_results[:,4] - box_results[:,4].min()
    #    box_results[:,4] = box_results[:,4] / box_results[:,4].max()
    #    print(box_results[:,4])
    
        box_results = torch.transpose( box_results, 0, 1 ).unsqueeze(0)
    
    
    # non-maxima suppression
        keep_ind = non_max_suppression(box_results,
            conf_thres=0,
            iou_thres=0,
            classes=None,
            agnostic=True,
            multi_label=False,
            labels=(),
            max_det=MAX_DET,
            nc=0,  # number of classes (optional)
            max_time_img=0.05,
            max_nms=MAX_DET,
            max_wh=7680,
            in_place=False,
            rotated=False,
        )
        box_results = box_results[:,:,keep_ind[0]].squeeze().transpose(0,1)
        mask_results = [ mask_results[i] for i in keep_ind[0] ]
    
#    print( cat_results.tolist() )
    
    with open("{f}/img_{id}.txt".format(f=out_dir, id=idx), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(box_results)):
            writer.writerow( box_results[i].tolist() + mask_results[i].flatten().tolist() )
    
    # Draw boxes on original image
    img1 = ImageDraw.Draw(img, 'RGBA')
    
    for i, box in enumerate(box_results):
        box = box[:4].type(torch.int)
        shape = [(box[0], box[1]), (box[2], box[3])]
        img1.rectangle(shape, outline="green", width=3)
#        print(mask_results[i].astype(int).flatten().tolist())
        mask = mask_results[i].astype(int).flatten().tolist()
        if len(mask) >= 6:
            color = (randint(0,255),randint(0,255),randint(0,255))
            img1.polygon(mask, fill=color+(125,), outline="blue")
            
    img.save( "{f}/unpatch.png".format(f=out_dir) )
    
    if objects_found:
        return [{"boxes":box_results[:,:4], "scores":box_results[:,4], "labels":box_results[:,5].int()}]
    else:
        return [{"boxes":[], "scores":[], "labels":[]}]

def load_annot(path):
    # Load the ground truth annotations
    bbf = csv.reader(open(path, newline=''), delimiter=',')
    bbs = []
    for row in bbf:
        if row[1] == "bb_x":
            continue
        
        bbs.append( [int(row[1]), int(row[2]), int(row[1])+int(row[3]), int(row[2])+int(row[4])] ) #xyxy
#        bbs.append( [ int(float(coor)) for coor in row[5:] ] )
    bbs = torch.Tensor(bbs)
    bbs = [{ "boxes":bbs, "labels":torch.zeros(bbs.shape[0], dtype=torch.int) }]
    return bbs



def model_inference(img_foldername="instance/uploads", out_foldername="uploads/output", ratio=0.7784):
    
    # move the sys args into parser
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--img_foldername", type=str, default="img")
#    parser.add_argument("--out_foldername", type=str, default="out")
#    parser.add_argument("--ratio", type=float, default=0.7784) #um per pixel
#    args = parser.parse_args()
    
    
    um_per_pixel = ratio
    patch_size = int( UM_PER_PATCH/um_per_pixel )
#    print("ps ")
#    print(patch_size)
    
    out_dir = out_foldername
    
    img_dir = img_foldername
#    img_dir = "/scratch/manne.sa/data/osteoclasts/drive/Dobutamine_Exp2/Images"
#    roi_dir = "/scratch/martin.br/roi_as_csv/dobu2/"
#    img_dir = args.img_foldername
#    img_dir = "data/Dobutamine/Dobutamine_Experiment_2/images/"
#    patch_dir = "data/unpatch_2"
#    patch_dir = os.path.join(args.out_foldername,"patches")
    img_files = os.listdir(img_dir)
    img_files = [f for f in img_files if f.endswith(".png")]
#    img_files = ["File_db9282bb_aaaa_4981_90f6_e30e003c2a4a.tif", "File_1b8e8852_5302_470c_ac19_fcac0c395f3a.tif"]
    
    model_path = "osteo_flask/models/best.pt"
    model = YOLO(model_path)
    
    # check if out_dir exists and create if it doesn't
#    if not os.path.exists( patch_dir ):
#        os.makedirs( patch_dir )
        
    # Load the ground truth annotations
#    bbf = csv.reader(open("data/roi_db9.csv", newline=''), delimiter=',')
#    bbs = []
#    for row in bbf:
#        if row[1] == "bb_x":
#            continue
#
#        bbs.append( [int(row[1]), int(row[2]), int(row[1])+int(row[3]), int(row[2])+int(row[4])] ) #xyxy
##        bbs.append( [ int(float(coor)) for coor in row[5:] ] )
#    bbs = torch.Tensor(bbs)
#    bbs = [{ "boxes":bbs, "labels":torch.zeros(bbs.shape[0], dtype=torch.int) }]
    
    
#    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True, max_detection_thresholds=[1,10,MAX_DET]).to("cpu")
    idx = 0
    for img_filename in img_files[:1]:
        # print img filename
        print(img_filename)
        
        img = Image.open( os.path.join(img_dir, img_filename) )
        
#        patch(img, patch_dir, idx, patch_size)
#        pred = unpatch(model, patch_dir, idx, img, patch_size, out_dir)
        pred = inference(model, idx, img, patch_size, out_dir)
        
#        roi_path = os.path.join( roi_dir, "roi_{id}.csv".format(id=img_filename.split("_")[1][:3]) )
#        bbs = load_annot(roi_path)
#        metric.update(pred, bbs)
        
        idx += 1
        
#    output = metric.compute()
#    pprint(output)
    
#    print(keep_ind.shape)
    
    return
    
def main(argv):
    model_inference()
    return

if __name__ == '__main__':
    main(sys.argv)
