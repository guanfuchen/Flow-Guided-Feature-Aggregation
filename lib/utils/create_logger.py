# coding=utf-8
# --------------------------------------------------------
# Flow-Guided-Feature-Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import time

# 创建日志打印类
def create_logger(root_output_path, cfg, image_set):
    # set up logger
    # 创建日志文件
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    # 读取cfg_name
    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)

    image_sets = [iset for iset in image_set.split('+')]
    # print('image_sets:', image_sets)
    # 通过以上综合设置log输出目录
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 返回logger日志记录器和最后的日志输出路径
    return logger, final_output_path

