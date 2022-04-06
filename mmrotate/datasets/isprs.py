# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
from venv import create
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import obb2poly_np, poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_map
from .builder import ROTATED_DATASETS
from xml.dom.minidom import Document
from PIL import Image

@ROTATED_DATASETS.register_module()
class ISPRSDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    ori_cls=('Passenger Ship', 'Motorboat', 'Fishing Boat',
    'Tugboat', 'other-ship', 'Engineering Ship', 'Liquid Cargo Ship', 
    'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck', 
    'Dump Truck', 'other-vehicle', 'Van', 'Trailer', 'Tractor', 
    'Excavator', 'Truck Tractor', 'Boeing737', 'Boeing747', 
    'Boeing777', 'Boeing787', 'ARJ21', 'C919', 'A220', 'A321', 
    'A330', 'A350', 'other-airplane', 'Baseball Field', 'Basketball Court', 
    'Football Field', 'Tennis Court', 'Roundabout', 'Intersection', 'Bridge')
    
    CLASSES = ('Passenger-Ship', 'Motorboat', 'Fishing-Boat', 
    'Tugboat', 'other-ship', 'Engineering-Ship', 'Liquid-Cargo-Ship',
    'Dry-Cargo-Ship', 'Warship', 'Small-Car', 'Bus', 
    'Cargo-Truck', 'Dump-Truck', 'other-vehicle', 'Van', 
    'Trailer', 'Tractor', 'Excavator', 'Truck-Tractor', 
    'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 
    'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350', 
    'other-airplane', 'Baseball-Field', 'Basketball-Court',
    'Football-Field', 'Tennis-Court', 'Roundabout', 'Intersection', 'Bridge')
    
    alias_dict={}
    for i in range(len(CLASSES)):
        alias_dict.update({CLASSES[i]:ori_cls[i]})

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(ISPRSDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.tif')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.tif'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.tif'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)


    def create_xml(self, img_id, in_dicts, out_path):
        doc = Document()
        root = doc.createElement('annotation')
        doc.appendChild(root)
        source_list = {'filename':img_id+'.tif', 'origin': 'GF2/GF3'}
        node_source = doc.createElement('source')
        for source in source_list:
            node_name = doc.createElement(source)
            node_name.appendChild(doc.createTextNode(source_list[source]))
            node_source.appendChild(node_name)
        root.appendChild(node_source)

        research_list = {'version': '1.0', 'provider': 'FAIR1M', 'author': 'Cyber',
                        'pluginname': 'FAIR1M', 'pluginclass': 'object detection', 'time': '2021-07-21'}
        node_research = doc.createElement('research')
        for research in research_list:
            node_name = doc.createElement(research)
            node_name.appendChild(doc.createTextNode(research_list[research]))
            node_research.appendChild(node_name)
        root.appendChild(node_research)

        img = Image.open(os.path.join('../FAIR1M/test/images',img_id+'.tif'))
        size_list = {'width': str(img.size[0]), 'height': str(img.size[1]), 'depth': '3'}
        node_size = doc.createElement('size')
        for size in size_list:
            node_name = doc.createElement(size)
            node_name.appendChild(doc.createTextNode(size_list[size]))
            node_size.appendChild(node_name)
        root.appendChild(node_size)

        node_objects = doc.createElement('objects')
        for cls_name in in_dicts.keys():

            for i in range(len(in_dicts[cls_name])):

                # if in_dicts[cls_name][i][8]>0.2:
                node_object = doc.createElement('object')
                object_fore_list = {'coordinate': 'pixel', 'type': 'rectangle', 'description': 'None'}
                for object_fore in object_fore_list:
                    node_name = doc.createElement(object_fore)
                    node_name.appendChild(doc.createTextNode(object_fore_list[object_fore]))
                    node_object.appendChild(node_name)

                node_possible_result = doc.createElement('possibleresult')
                node_name = doc.createElement('name')
                node_name.appendChild(doc.createTextNode(cls_name))
                node_possible_result.appendChild(node_name)

                node_probability = doc.createElement('probability')
                node_probability.appendChild(doc.createTextNode(str(in_dicts[cls_name][i][8])))
                node_possible_result.appendChild(node_probability)

                node_object.appendChild(node_possible_result)

                node_points = doc.createElement('points')

                for j in range(4):
                    node_point = doc.createElement('point')

                    text = '{},{}'.format(in_dicts[cls_name][i][int(0+2*j)], in_dicts[cls_name][i][int(1+2*j)])
                    node_point.appendChild(doc.createTextNode(text))
                    node_points.appendChild(node_point)
                    
                node_point = doc.createElement('point')
                text = '{},{}'.format(in_dicts[cls_name][i][0], in_dicts[cls_name][i][1])
                node_point.appendChild(doc.createTextNode(text))
                node_points.appendChild(node_point)
                node_object.appendChild(node_points)
                node_objects.appendChild(node_object)

        root.appendChild(node_objects)

        # 开始写xml文档
        filename = os.path.join(out_path, img_id + '.xml')
        fp = open(filename, 'w', encoding='utf-8')
        doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()



    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if not osp.exists(out_folder):
            os.makedirs(out_folder)

        for img_id, dets_per_cls in zip(id_list, dets_list):
            result_dict={}
            for cls_name, dets in zip(self.CLASSES, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                result_dict.update({self.alias_dict[cls_name]:bboxes})

            self.create_xml(img_id,result_dict,out_folder)

        return None

    def format_results(self, results, submission_dir='test', nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
