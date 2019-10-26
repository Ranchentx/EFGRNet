import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
# from models.model_builder import SSD
from models.model_builder_vgg import SSD

# from models.model_builder_resnet import SSD


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        '--weights',
        # default='/media/jnie/Storage/iccv_weights/efrgnet_vgg_epoch_320.pth',
        default='/media/jnie/Storage/iccv_weights/efrgnet_vgg_epoch_512.pth',
        type=str,
        help='Trained state_dict file path to open')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--save_folder',
        default='eval/',
        type=str,
        help='File path to save results')
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')
    args = parser.parse_args()
    return args


def eval_net(val_dataset,
             val_loader,
             net,
             detector,
             cfg,
             transform,
             max_per_image=300,
             thresh=0.01,
             batch_size=1):
    net.eval()
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    eval_save_folder = "./eval/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    det_file = os.path.join(eval_save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        val_dataset.evaluate_detections(all_boxes, eval_save_folder)
        return

    # img_idexes = val_dataset.image_indexes  #coco
    img_idexes = val_dataset.ids  #voc
    total_times = 0
    network_times = 0
    total_nms_times = 0
    total_forward_times = 0

    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():

            x = imgs
            x = x.cuda()
            torch.cuda.synchronize()
            t1 = time.time()
            output= net(x)

            torch.cuda.synchronize()
            t4 = time.time()
            boxes, scores = detector.forward(output)

            # idx = np.where(scores == scores[:,:,1:].max())
            torch.cuda.synchronize()
            t2 = time.time()

            for k in range(boxes.size(0)):
                i = idx * batch_size + k
                boxes_ = boxes[k]
                scores_ = scores[k]
                boxes_ = boxes_.cpu().numpy()
                scores_ = scores_.cpu().numpy()
                img_wh = img_info[k]
                scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
                boxes_ *= scale
                for j in range(1, num_classes):
                    inds = np.where(scores_[:, j] > thresh)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes_[inds]
                    c_scores = scores_[inds, j]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(
                                            np.float32, copy=False)
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=False)
                    # keep = soft_nms(c_dets, Nt=0.45, method=2)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets

            torch.cuda.synchronize()
            t3 = time.time()
            detect_time = t2 - t4
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.6f}s {:.6f}s {:.3f}s'.format(
                    i + 1, num_images, forward_time, detect_time, nms_time))
            network_times += (t4 - t1)
            total_times += (t3 - t1)
            total_nms_times += nms_time
            total_forward_times += (t2-t1)

    print("detect time: ", time.time() - st)
    print("net time: ", network_times/5000.0)
    print("avg time: ", total_times/5000.0)
    print("nms time: ", total_nms_times/5000.0)
    print("forward time: ", total_forward_times/5000.0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)


def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    bgr_means = cfg.TRAIN.BGR_MEAN
    dataset_name = cfg.DATASETS.DATA_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    num_workers = args.num_workers
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 200
    else:
        trainvalDataset = COCODetection
        top_k = 300
    dataroot = cfg.DATASETS.DATAROOT
    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG

    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)

    checkpoint = torch.load(args.weights)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    detector = Detect(cfg)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))

    val_dataset = trainvalDataset(dataroot, valSet, ValTransform)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate)
    top_k = 300
    thresh = cfg.TEST.CONFIDENCE_THRESH
    eval_net(
        val_dataset,
        val_loader,
        net,
        detector,
        cfg,
        ValTransform,
        top_k,
        thresh=thresh,
        batch_size=batch_size)


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
