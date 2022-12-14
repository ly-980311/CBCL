# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on single image.

Example:


```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/image_demo.py \
    demo/demo.jpg \
    configs/oriented_rcnn/orcnn_r50_fpn_1x_ISPRS_3s_le90.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
    demo/vis.jpg
```
"""  # nowq

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

import mmrotate  # noqa: F401


def main():
    """Test a single image."""
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('output', help='Output image')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    model.show_result(
        args.img, result, score_thr=args.score_thr, out_file=args.output)


if __name__ == '__main__':
    main()
