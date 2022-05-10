import math
import numpy as np
import os
import pdb
from mmcv import Config
import mmcv
import DOTA_devkit.polyiou as polyiou
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import build_dataset
import argparse
import json


def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def obb2poly(obboxes):
    center, w, h, theta, confidence = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, -h / 2 * Cos], axis=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2

    a=1
    return np.concatenate([point1, point2, point3, point4, confidence], axis=-1)


class DetectorModel:
    def __init__(self,
                 config_file,
                 checkpoint_file,
                 device):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = build_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device=device)

    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]

        for i in range(int(width / slide_w + 1)):
            for j in range(int(height / slide_h) + 1):
                subimg = np.zeros((hn, wn, channel))
                chip = img[j * slide_h:j * slide_h + hn, i * slide_w:i * slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip

                chip_detections = inference_detector(self.model, subimg)

                temp_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
                for cls_id, name in enumerate(self.classnames):
                    poly_bbox = []
                    if chip_detections[cls_id].shape[0] != 0:
                        for xywht_bbox in chip_detections[cls_id]:
                            poly_bbox.append(obb2poly(xywht_bbox))
                        temp_detections[cls_id] = np.concatenate((temp_detections[cls_id], poly_bbox))
                        temp_detections[cls_id][:, :8][:, ::2] = temp_detections[cls_id][:, :8][:, ::2] + i * slide_w
                        temp_detections[cls_id][:, :8][:, 1::2] = temp_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                        # import pdb;pdb.set_trace()
                        try:
                            total_detections[cls_id] = np.concatenate((total_detections[cls_id],
                                                                       temp_detections[cls_id]))
                        except:
                            import pdb
                            pdb.set_trace()
        # nms
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.5)
            total_detections[i] = total_detections[i][keep]
        return total_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # local test
    parser.add_argument("--input_dir", default='/home/liyan/FAIR1M/test/images/', help="input path", type=str)
    parser.add_argument("--output_dir", default='output_path/ISPRS_obb_r101_3s_dcn_48e_st/', help="output path", type=str)
    parser.add_argument('--pid_index', default=1, help='pid', type=int)
    parser.add_argument('--num_pool', default=8, help='pid', type=int)

    parser.add_argument('--config', default='work_dirs/ISPRS_obb_r101_3s_dcn_48e/ISPRS_obb_r101_3s_dcn_48e.py',
                        help='config file', type=str)
    parser.add_argument('--checkpoint', default='work_dirs/ISPRS_obb_r101_3s_dcn_48e/epoch_48.pth',
                        help='checkpoint file', type=str)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', default=0, help='bbox score threshold', type=float)
    args = parser.parse_args()

    input_img_list = os.listdir(args.input_dir)
    num_img = len(input_img_list)
    list_clip = list(range(0, num_img, int(num_img / args.num_pool) + 1))
    input_img_lists = []
    # if args.num_pool > 1:
    for i in range(len(list_clip)):
        try:
            input_img_lists.append(input_img_list[list_clip[i]:list_clip[i + 1]])
        except:
            input_img_lists.append(input_img_list[list_clip[i]:])
    # else:
    #     input_img_lists.append(input_img_list)

    output_dicts = []
    classes = ['Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'other-ship',
               'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship',
               'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'other-vehicle',
               'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor',
               'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21',
               'C919', 'A220', 'A321', 'A330', 'A350', 'other-airplane',
               'Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court',
               'Roundabout', 'Intersection', 'Bridge']
    # classes = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21',
    #            'C919', 'A220', 'A321', 'A330', 'A350', 'other-airplane']
    threshold = args.score_thr
    count = 0
    model = DetectorModel(args.config, args.checkpoint, args.device)
    slide_size = (512, 512)
    chip_size = (1024, 1024)
    total = len(input_img_lists[args.pid_index])
    print('pool {} will test {} image(s)'.format(args.pid_index, total))
    for input_img in input_img_lists[args.pid_index]:
        input_path = os.path.join(args.input_dir, input_img)
        count += 1
        print('pid {} num {}/{} \ttest {}'.format(args.pid_index, count, total, input_img))
        result = model.inference_single(input_path, slide_size, chip_size)
        output_dict = {}
        output_dict.update({'image_name': input_img})
        labels = []
        for class_id, bbox_result in enumerate(result):
            if bbox_result.shape[0] != 0:
                for index in range(bbox_result.shape[0]):
                    if bbox_result[index, 8] > threshold:
                        x1 = bbox_result[index, 0]
                        y1 = bbox_result[index, 1]
                        x2 = bbox_result[index, 2]
                        y2 = bbox_result[index, 3]
                        x3 = bbox_result[index, 4]
                        y3 = bbox_result[index, 5]
                        x4 = bbox_result[index, 6]
                        y4 = bbox_result[index, 7]
                        points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        category_id = classes[class_id]
                        confidence = bbox_result[index, 8]
                        labels.append({'points': points, 'category_id': category_id, 'confidence': confidence})
        output_dict.update({'labels': labels})
        output_dicts.append(output_dict)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = '{}results_{}.json'.format(args.output_dir, args.pid_index)
    mmcv.dump(output_dicts, output_path, indent=True)
