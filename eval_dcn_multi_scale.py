#-------------------------------
# This code refers to https://github.com/sfzhang15/RefineDet.git
# ------------------------------
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
from data import COCODetection, VOCDetection,detection_collate, BaseTransform, BaseTransform_img, BaseTransform_ration
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
from models.model_builder_resnet import SSD

import cv2


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        '--weights',
        default='./weights/refine_res_epoch_160_512.pth',
        type=str,
        help='Trained state_dict file path to open')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--save_folder',
        default='eval_ms/',
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


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    net.eval()
    num_images = len(testset)
    num_classes = 81
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()
        out = net(x)      # forward pass

        boxes, scores = detector.forward(out)
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=False)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if i % 20 == 0:
            print('im_detect: {:d}/{:d}'
                .format(i + 1, num_images))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


def im_detect(net, detector, cuda, img, targe_size):
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    transform = BaseTransform_img(targe_size, (104, 117, 123), (2, 0, 1))
    num_classes = 81
    thresh = 0.001
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)  # forward pass
    boxes, scores = detector.forward(out)
    boxes = boxes[0]
    scores = scores[0]

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    det_tempolate = []
    for j in range(1, num_classes):

        keep_inds = np.where(scores[:,j] >= 0.01)[0]

        c_bboxes = boxes[keep_inds]
        c_scores = scores[keep_inds, j]
        c_dets = np.column_stack((c_bboxes, c_scores))
        # keep = nms(c_dets, 0.45, force_cpu=False)
        # keep = keep[:50]
        # c_dets = c_dets[keep, :]
        # num = np.ones(len(keep))
        num = np.ones(c_dets.shape[0])
        det_tempolate.append(np.insert(c_dets, 5, values=num *j, axis=1))
    dets = np.concatenate(det_tempolate)
    return dets

def im_detect_ratio(net, detector, cuda, img, targe_size1, targe_size2):
    im_orig = img.astype(np.float32, copy=True)
    if im_orig.shape[0] < im_orig.shape[1]:
        tmp = targe_size1
        targe_size1 = targe_size2
        targe_size2 = tmp

    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    transform = BaseTransform_ration([float(targe_size2)/float(img.shape[1]),float(targe_size1)/float(img.shape[0])], (104, 117, 123), (2, 0, 1))
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)  # forward pass
    boxes, scores = detector.forward(out)
    boxes = boxes[0]
    scores = scores[0]

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    #
    # keep_inds = np.where(scores[:, 1] >= 0.01)[0]
    # c_bboxes = boxes[keep_inds]
    # c_scores = scores[keep_inds, 1]
    # c_dets = np.column_stack((c_bboxes, c_scores))
    # keep = nms(c_dets, 0.45, force_cpu=False)
    # keep = keep[:50]
    # c_dets = c_dets[keep, :]
    # num = np.ones(len(keep))
    # dets = np.insert(c_dets, 5, values=num, axis=1)
    # return dets
    num_classes = 81
    thresh = 0.001
    det_tempolate = []
    for j in range(1, num_classes):

        keep_inds = np.where(scores[:,j] >= 0.01)[0]

        c_bboxes = boxes[keep_inds]
        c_scores = scores[keep_inds, j]
        c_dets = np.column_stack((c_bboxes, c_scores))
        # keep = nms(c_dets, 0.45, force_cpu=False)
        # keep = keep[:50]
        # c_dets = c_dets[keep, :]
        # num = np.ones(len(keep))
        num = np.ones(c_dets.shape[0])
        det_tempolate.append(np.insert(c_dets, 5, values=num * j, axis=1))
    dets = np.concatenate(det_tempolate)
    return dets

def flip_im_detect(net, detector, cuda, img, targe_size):
    im_f = cv2.flip(img, 1)
    det_f =im_detect(net, detector, cuda, im_f, targe_size)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = img.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = img.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t

def flip_im_detect_ratio(net, detector, cuda, im, targe_size1, targe_size2):
    im_f = cv2.flip(im, 1)
    det_f = im_detect_ratio(net, detector, cuda, im_f, targe_size1, targe_size2)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t

def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets


def soft_bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= 0.001)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


def test_net_multiscale_512(save_folder, net, detector, cuda, testset):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    net.eval()
    num_images = len(testset)
    num_classes = 81
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    targe_size = int(512)

    for i in range(num_images):
        im = testset.pull_image(i)
        det0 = im_detect(net, detector, cuda, im, targe_size)
        det0_f = flip_im_detect(net, detector, cuda, im, targe_size)
        det0 = np.row_stack((det0, det0_f))

        # det_r = im_detect_ratio(net, detector, cuda, im, targe_size, int(0.75 * targe_size))
        # det_r_f = flip_im_detect_ratio(net, detector, cuda, im, targe_size, int(0.75 * targe_size))
        # det_r = np.row_stack((det_r, det_r_f))


        # shrink: only detect big object
        det_s1 = im_detect(net, detector, cuda, im, int(0.5 * targe_size))
        det_s1_f = flip_im_detect(net, detector, cuda, im, int(0.5 * targe_size))
        det_s1 = np.row_stack((det_s1, det_s1_f))

        det_s2 = im_detect(net, detector, cuda, im, int(0.75 * targe_size))
        det_s2_f = flip_im_detect(net, detector, cuda, im, int(0.75 * targe_size))
        det_s2 = np.row_stack((det_s2, det_s2_f))

        # --------------jnie-----------------------
        index = np.where(np.maximum(det_s2[:, 2] - det_s2[:, 0] + 1, det_s2[:, 3] - det_s2[:, 1] + 1) > 64)[0]
        det_s2 = det_s2[index, :]
        # ------------------------------------------------------------------------------------------------------



        # #enlarge: only detect small object
        det3 = im_detect(net, detector, cuda, im, int(1.75 * targe_size))
        det3_f = flip_im_detect(net, detector, cuda, im, int(1.75 * targe_size))
        det3 = np.row_stack((det3, det3_f))
        index = np.where(np.minimum(det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 128)[0]
        det3 = det3[index, :]

        det4 = im_detect(net, detector, cuda, im, int(1.5 * targe_size))
        det4_f = flip_im_detect(net, detector, cuda, im, int(1.5 * targe_size))
        det4 = np.row_stack((det4, det4_f))
        index = np.where(np.minimum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1) < 192)[0]
        det4 = det4[index, :]

        # det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4))
        # det = np.row_stack((det0, det_r, det_s1, det3))
        # det =det0
        # coco
        det5 = im_detect(net, detector, cuda, im, int(1.25 * targe_size))
        det5_f = flip_im_detect(net, detector, cuda, im, int(1.25 * targe_size))
        det5 = np.row_stack((det5, det5_f))
        index = np.where(np.minimum(det5[:, 2] - det5[:, 0] + 1, det5[:, 3] - det5[:, 1] + 1) < 224)[0]
        det5 = det5[index, :]

        det6 = im_detect(net, detector, cuda, im, int(2 * targe_size))
        det6_f = flip_im_detect(net, detector, cuda, im, int(2 * targe_size))
        det6 = np.row_stack((det6, det6_f))
        index = np.where(np.minimum(det6[:, 2] - det6[:, 0] + 1, det6[:, 3] - det6[:, 1] + 1) < 96)[0]
        det6 = det6[index, :]

        det7 = im_detect(net, detector, cuda, im, int(2.25 * targe_size))
        det7_f = flip_im_detect(net, detector, cuda, im, int(2.25 * targe_size))
        det7 = np.row_stack((det7, det7_f))
        index = np.where(np.minimum(det7[:, 2] - det7[:, 0] + 1, det7[:, 3] - det7[:, 1] + 1) < 64)[0]
        det7 = det7[index, :]
        det = np.row_stack((det0, det_s1, det_s2, det3, det4, det5, det6, det7))
        # det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4, det5))
        # det = np.row_stack((det0, det_r, det_s1, det_s2, det4, det5))
        # det = np.row_stack((det0, det_s1, det_s2, det4, det5))
        # det = det0


        for j in range(1, num_classes):
            inds = np.where(det[:, -1] == j)[0]
            # inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                # cls_dets = bbox_vote(cls_dets)
                # cls_dets = soft_bbox_vote(cls_dets)

                keep = soft_nms(cls_dets, Nt=0.45, method=2)
                keep = keep[:100]
                cls_dets = cls_dets[keep, :]

                all_boxes[j][i] = cls_dets



        if i % 20 == 0:
            print('im_detect: {:d}/{:d}'
                  .format(i + 1, num_images))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)




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

    if True:
        net = net.cuda()
        cudnn.benchmark = True

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
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, "val")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate)
    top_k = 200
    thresh = cfg.TEST.CONFIDENCE_THRESH

    rgb_means = (104, 117, 123)
    test_net_multiscale_512(save_folder, net, detector, True, val_dataset)

if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
