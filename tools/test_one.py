# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import cv2
import numpy as np
import gradio as gr
    
import mmcv
import torch
from mmengine import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from uuid import uuid4


import json

def dirty_xray(img):
    # 1. save a sample dataset to tmp
    # 2. run the test_one.py
    # 3. load visualization from tmp
    # 4. return the visualization
    img_id = str(uuid4())
    cv2.imwrite(f"./tmp/{img_id}_orig.jpg", img)
    
    cv2.imwrite("./data/test_img/sample.jpg", img)
    h,w = img.shape[:2]
    temp_anno = {
        "annotations": [],
        "categories": [
        {"id": 1,"name": "Baton"},
        {"id": 2,"name": "Pliers"},
        {"id": 3,"name": "Hammer"},
        {"id": 4,"name": "Powerbank"},
        {"id": 5,"name": "Scissors"},
        {"id": 6,"name": "Wrench"},
        {"id": 7,"name": "Gun"},
        {"id": 8,"name": "Bullet"},
        {"id": 9,"name": "Sprayer"},
        {"id": 10,"name": "HandCuffs"},
        {"id": 11,"name": "Knife"},
        {"id": 12,"name": "Lighter"}
        ],
    "images": [
        {
            "file_name": "sample.jpg",
            "height": h,
            "id": 0,
            "width": w
        }
    ],
    "info": "spytensor created",
    "license": [
        "license"
    ]}
    with open("./data/test_anno/test_one.json", 'w') as fp:
        json.dump(temp_anno, fp, sort_keys=True, indent=4, ensure_ascii=False)
    
    

    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    args = parser.parse_args()
    
    config_path = './configs/cascade_mask_rcnn_r101_with_R0R1_single.py'
    args.checkpoint = '/Users/htplex/Desktop/data_new/datasets/PIDray/model_weights/cascade_mask_rcnn_r101_with_R0R1.pth'
    args.work_dir = None
    args.out = None
    # args.fuse_conv_bn = True
    args.format_only = True
    args.eval = None
    args.show = True
    args.show_dir = './tmp/'
    args.show_score_thr = 0.3
    args.gpu_collect = True
    args.tmpdir = None
    args.options = None
    args.eval_options = None
    

    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.device = 'cpu'
    # init distributed env first, since logger depends on the dist info.

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    # in case the test dataset is concatenated

    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
 

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()


    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    from pprint import pprint
    pprint(dataset)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        
        


    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    
    
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
    
    


    rank, _ = get_dist_info()
    
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)
        metric_dict = dict(config=config_path, metric=metric)
        if args.work_dir is not None and rank == 0:
            mmcv.dump(metric_dict, json_file)
            
    result = cv2.imread("./tmp/sample.jpg")
    cv2.imwrite(f"./tmp/{img_id}_result.jpg", img)

            
    return result
    


if __name__ == '__main__':
    # local demo
    # img = cv2.imread("/Users/htplex/Desktop/data_new/datasets/PIDray/test/xray_hard00069.png")
    # result = dirty_xray(img)
    # print(result.shape)
    # cv2.imshow("result", result)
    
    # gradio demo


    demo = gr.Interface(dirty_xray, gr.Image(), "image")
    # demo.launch(server_name="0.0.0.0", server_port=7878)
    demo.launch(share=True)
