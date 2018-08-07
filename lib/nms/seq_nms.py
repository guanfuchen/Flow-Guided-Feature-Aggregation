# coding=utf-8
# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified byYuqing Zhu, Xizhou Zhu
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np

import profile
import cv2
import time
import copy
import cPickle as pickle
import os
import numpy as np

CLASSES = ('__background__',
           'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
           'car', 'cattle', 'dog', 'domestic cat', 'elephant', 'fox',
           'giant panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
           'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')

           
NMS_THRESH = 0.3
IOU_THRESH = 0.5
MAX_THRESH=1e-2


def createLinks(dets_all):
    """
    创建相邻帧的连接构造图，连接规则是相邻帧的目标框IOU>0.5的目标框进行连接
    :param dets_all:
    :return:
    """
    links_all = []

    # 帧数目dets_all的shape为（#CLASSES, T） 其中T为帧数目，#CLASSES为类别数目，这里为（30,144），也就是30个类别，144帧
    frame_num = len(dets_all[0])
    # 总共类别len(CLASSES)（包括目标类），实际类别为len(CLASSES)-1
    cls_num = len(CLASSES) - 1
    # 遍历规则是对于每一个类别遍历，然后在帧间遍历
    for cls_ind in range(cls_num):
        links_cls = []
        for frame_ind in range(frame_num - 1):
            # 当前帧和后一帧检测类别的检测结果
            dets1 = dets_all[cls_ind][frame_ind]
            dets2 = dets_all[cls_ind][frame_ind + 1]
            # 对应检测结果的box_num
            box1_num = len(dets1)
            box2_num = len(dets2)
            
            if frame_ind == 0:
                # 第一帧的情况下，面积计算为第一帧的面积box格式为（xmin, ymin，xmax，ymax）
                areas1 = np.empty(box1_num)
                for box1_ind, box1 in enumerate(dets1):
                    areas1[box1_ind] = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            else:
                # 记录上一帧的面积
                areas1 = areas2

            areas2 = np.empty(box2_num)
            for box2_ind, box2 in enumerate(dets2):
                areas2[box2_ind] = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            # 连接帧
            links_frame = []
            for box1_ind, box1 in enumerate(dets1):
                area1 = areas1[box1_ind]
                # 当前帧和后一帧所有的IOU计算
                x1 = np.maximum(box1[0], dets2[:, 0])
                y1 = np.maximum(box1[1], dets2[:, 1])
                x2 = np.minimum(box1[2], dets2[:, 2])
                y2 = np.minimum(box1[3], dets2[:, 3])
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                # 计算IOU
                ovrs = inter / (area1 + areas2 - inter)
                # 连接所有IOU>IOU_THRESH的box
                links_box = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if
                             ovr >= IOU_THRESH]
                # 连接满足连接IOU条件的links，没有则为[]
                links_frame.append(links_box)
            links_cls.append(links_frame)
        links_all.append(links_cls)
    return links_all


def maxPath(dets_all, links_all):
    """
    查找links_all中最大dets置信度的路径
    :param dets_all: 所有类别在每一帧中的检测框
    :param links_all: 相邻帧bbox IOU>0.5的连接构造图
    :return:
    """
    for cls_ind, links_cls in enumerate(links_all):
        max_begin = time.time()
        delete_sets=[[]for i in range(0,len(dets_all[0]))]
        delete_single_box=[]
        dets_cls = dets_all[cls_ind]

        num_path=0
        # compute the number of links
        sum_links=0
        for frame_ind, frame in enumerate(links_cls):
            for box_ind,box in enumerate(frame):
                sum_links+=len(box)

        while True:

            num_path+=1

            rootindex, maxpath, maxsum = findMaxPath(links_cls, dets_cls,delete_single_box)

            if (maxsum<MAX_THRESH or sum_links==0 or len(maxpath) <1):
                break
            if (len(maxpath)==1):
                delete=[rootindex,maxpath[0]]
                delete_single_box.append(delete)
            # 重新打分
            rescore(dets_cls, rootindex, maxpath, maxsum)
            t4=time.time()
            # 删除Link，IOU大于一定的NMS抑制
            delete_set,num_delete=deleteLink(dets_cls, links_cls, rootindex, maxpath, NMS_THRESH)
            sum_links-=num_delete
            for i, box_ind in enumerate(maxpath):
                delete_set[i].remove(box_ind)
                delete_single_box.append([[rootindex+i],box_ind])
                for j in delete_set[i]:
                    dets_cls[i+rootindex][j]=np.zeros(5)
                delete_sets[i+rootindex]=delete_sets[i+rootindex]+delete_set[i]

        for frame_idx,frame in enumerate(dets_all[cls_ind]):

            a=range(0,len(frame))
            keep=list(set(a).difference(set(delete_sets[frame_idx])))
            dets_all[cls_ind][frame_idx]=frame[keep,:]


    return dets_all


def findMaxPath(links,dets,delete_single_box):

    len_dets=[len(dets[i]) for i in xrange(len(dets))]
    max_boxes=np.max(len_dets)
    num_frame=len(links)+1
    a=np.zeros([num_frame,max_boxes])
    new_dets=np.zeros([num_frame,max_boxes])
    for delete_box in delete_single_box:
        new_dets[delete_box[0],delete_box[1]]=1
    if(max_boxes==0):
        max_path=[]
        return 0,max_path,0

    b=np.full((num_frame,max_boxes),-1)
    for l in xrange(len(dets)):
        for j in xrange(len(dets[l])):
            if(new_dets[l,j]==0):
                a[l,j]=dets[l][j][-1]



    for i in xrange(1,num_frame):
        l1=i-1
        for box_id,box in enumerate(links[l1]):
            for next_box_id in box:

                weight_new=a[i-1,box_id]+dets[i][next_box_id][-1]
                if(weight_new>a[i,next_box_id]):
                    a[i,next_box_id]=weight_new
                    b[i,next_box_id]=box_id

    i,j=np.unravel_index(a.argmax(),a.shape)

    maxpath=[j]
    maxscore=a[i,j]
    while(b[i,j]!=-1):

            maxpath.append(b[i,j])
            j=b[i,j]
            i=i-1


    rootindex=i
    maxpath.reverse()
    return rootindex, maxpath, maxscore


def rescore(dets, rootindex, maxpath, maxsum):
    """
    重打分
    :param dets:
    :param rootindex:
    :param maxpath:
    :param maxsum:
    :return:
    """
    # 平均重打分
    newscore = maxsum / len(maxpath)

    for i, box_ind in enumerate(maxpath):
        # 对于这些最大path的打分为较高的score
        dets[rootindex + i][box_ind][4] = newscore


def deleteLink(dets, links, rootindex, maxpath, thesh):

    delete_set=[]
    num_delete_links=0

    for i, box_ind in enumerate(maxpath):
        areas = [(box[2] - box[0] + 1) * (box[3] - box[1] + 1) for box in dets[rootindex + i]]
        area1 = areas[box_ind]
        box1 = dets[rootindex + i][box_ind]
        x1 = np.maximum(box1[0], dets[rootindex + i][:, 0])
        y1 = np.maximum(box1[1], dets[rootindex + i][:, 1])
        x2 = np.minimum(box1[2], dets[rootindex + i][:, 2])
        y2 = np.minimum(box1[3], dets[rootindex + i][:, 3])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h

        ovrs = inter / (area1 + areas - inter)
        #saving the box need to delete
        deletes = [ovr_ind for ovr_ind, ovr in enumerate(ovrs) if ovr >= 0.3]
        delete_set.append(deletes)

        #delete the links except for the last frame
        if rootindex + i < len(links):
            for delete_ind in deletes:
                num_delete_links+=len(links[rootindex+i][delete_ind])
                links[rootindex + i][delete_ind] = []

        if i > 0 or rootindex > 0:

            #delete the links which point to box_ind
            for priorbox in links[rootindex + i - 1]:
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)
                        num_delete_links+=1

    return delete_set,num_delete_links

def seq_nms(dets):
    """"
    seq_nms：序列非极大值抑制，输入为检测框结果（包含检测框bbox和scores）
    """
    # print('dets:', dets)
    print('dets.shape:', np.array(dets).shape)
    # 创建相邻帧的连接构造图
    links = createLinks(dets)
    # 动态规划算法求解最大dets之和的连接路径
    dets=maxPath(dets, links)
    return dets

