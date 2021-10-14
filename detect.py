import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

import json


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    # filter removes empty strings (such as last line)
    return list(filter(None, names))


def detect(img_dir_path, idx=-1):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        model.load_state_dict(torch.load(
            weights[0], map_location=device)['model'])
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights[0])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    save_img = True
    dataset = LoadImages(img_dir_path, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    if opt.dataset == 'jackson':
        dataset.skip(idx * 100000)

    # Results
    res = dict()

    img_idx = 0
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        # jackson dataset only
        if opt.dataset == 'jackson' and img_idx == 100000:
            break

        print(path, end='')

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' %
                                                        dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                res[img_idx] = dict()
                res[img_idx]['class'] = []
                res[img_idx]['score'] = []
                for *xyxy, conf, cls in det:
                    conf_v, cls_v = conf.item(), cls.item()
                    res[img_idx]['class'] += [cls_v]
                    res[img_idx]['score'] += [conf_v]
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s' % (names[int(cls)])
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        img_idx += 1

        # Stream results
        # if view_img:
        #     cv2.imshow(p, im0)
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         raise StopIteration

        # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'images':
        #         cv2.imwrite(save_path, im0)
        #     else:
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer

        #             fourcc = 'mp4v'  # output video codec
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #         vid_writer.write(im0)

    # if save_txt or save_img:
    #     print('Results saved to %s' % Path(out))
    #     if platform == 'darwin' and not opt.update:  # MacOS
    #         os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    if 'p7' in opt.weights[0]:
        save_name = 'res-4.json'
    elif 'p6' in opt.weights[0]:
        save_name = 'res-3.json'
    elif 'p5' in opt.weights[0]:
        save_name = 'res-2.json'
    elif 'csp' in opt.weights[0]:
        save_name = 'res-1.json'
    elif 'tiny' in opt.weights[0]:
        save_name = 'res-0.json'
    else:
        raise Exception('Undefined model {}'.format(opt.weights))

    if opt.dataset == 'jackson':
        print(os.path.join(img_dir_path, '{}'.format(idx), save_name))
        with open(os.path.join(img_dir_path, '{}'.format(idx), save_name), 'w') as f:
            json.dump(res, f)
    else:
        with open(os.path.join(img_dir_path, save_name), 'w') as f:
            json.dump(res, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov4-csp.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--cfg', type=str,
                        default='models/yolov4-csp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str,
                        default='data/coco.names', help='*.cfg path')

    parser.add_argument('--dataset', required=True, help='Dataset')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            if opt.dataset != 'jackson':
                if opt.dataset == 'virat':
                    root_dir = '/data/jiashenc/virat/'
                    for dir_path in os.listdir(root_dir):
                        img_dir_path = os.path.join(root_dir, dir_path)
                        if os.path.isfile(img_dir_path) or 'VIRAT' not in img_dir_path:
                            continue
                        detect(img_dir_path)

                elif opt.dataset == 'bdd':
                    root_dir = '/data/jiashenc/bdd/bdd100k/videos/test'
                    for dir_path in os.listdir(root_dir):
                        img_dir_path = os.path.join(root_dir, dir_path)
                        if os.path.isfile(img_dir_path):
                            continue
                        detect(img_dir_path)

                elif opt.dataset == 'ua-detrac':
                    root_dir = '/data/jiashenc/ua_detrac/test'
                    for dir_path in os.listdir(root_dir):
                        img_dir_path = os.path.join(root_dir, dir_path)
                        if os.path.isfile(img_dir_path):
                            continue
                        detect(img_dir_path)

            else:
                root_dir = '/data/jiashenc/jackson'
                for i in range(0, 11):
                    detect(root_dir, i)
