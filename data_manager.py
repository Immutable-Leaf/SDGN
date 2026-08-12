from __future__ import print_function, absolute_import
import os
import numpy as np
import random


def _parse_sysu_img_path(img_path):
    """
    Robustly parse SYSU-MM01 image path.

    Expected path format:
        .../SYSU-MM01/cam3/0001/xxxx.jpg

    return:
        camid: 3
        pid: 1
    """
    img_path = os.path.normpath(img_path)
    parts = img_path.split(os.sep)

    # ... / camX / pid / img_name
    cam_name = parts[-3]   # cam3, cam6, cam1, ...
    pid_name = parts[-2]   # 0001, 0045, ...

    if not cam_name.startswith('cam'):
        raise ValueError('Unexpected SYSU path format: {}'.format(img_path))

    camid = int(cam_name.replace('cam', ''))
    pid = int(pid_name)

    return camid, pid


def _list_images(img_dir):
    """
    List image files in a directory.
    """
    img_files = []
    for name in os.listdir(img_dir):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_files.append(os.path.join(img_dir, name))
    return sorted(img_files)


def process_query_sysu(data_path, mode='all', relabel=False):
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']
    else:
        raise ValueError('Unsupported SYSU mode: {}'.format(mode))

    file_path = os.path.join(data_path, 'exp/test_id.txt')

    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ['%04d' % x for x in ids]

    for pid in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                new_files = _list_images(img_dir)
                files_ir.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []

    for img_path in files_ir:
        camid, pid = _parse_sysu_img_path(img_path)
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False):
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
    else:
        raise ValueError('Unsupported SYSU mode: {}'.format(mode))

    file_path = os.path.join(data_path, 'exp/test_id.txt')

    files_rgb = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ['%04d' % x for x in ids]

    for pid in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                new_files = _list_images(img_dir)
                if len(new_files) > 0:
                    files_rgb.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []

    for img_path in files_rgb:
        camid, pid = _parse_sysu_img_path(img_path)
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal=1):
    # In your current code:
    # modal == 1: thermal / infrared
    # modal == 2: visible
    if modal == 2:
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 1:
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    else:
        raise ValueError('Unsupported RegDB modal: {}'.format(modal))

    with open(input_data_path, 'r') as f:
        data_file_list = f.read().splitlines()

    file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
    file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)