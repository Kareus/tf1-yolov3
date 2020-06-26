from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import cv2
import sys
import random

PY_VERSION = sys.version_info[0]
iter_cnt = 0

def _parse_tfrecord(data):
    example = tf.train.Example()
    example.ParseFromString(data)
    features = example.features.feature
    return features

def parse_tfrecord(data):
    # tfrecord parser for TFRecordDataset (raw data)
    features = _parse_tfrecord(data)
    index = features['index'].int64_list.value[0]
    encoded_image = np.frombuffer(features['image'].bytes_list.value[0], dtype = np.uint8)
    width = features['width'].int64_list.value[0]
    height = features['height'].int64_list.value[0]
    boxes = features['boxes'].int64_list.value

    assert len(boxes) % 5 == 0, 'Annotation error occured in box array.'
    box_cnt = len(boxes) // 5

    aligned_boxes = []
    labels = []

    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(boxes[i * 5]), float(boxes[i * 5 + 1]), float(boxes[i * 5 + 2]), float(boxes[i * 5 + 3]), float(boxes[i * 5 + 4]) ## do we need to change int to float? is there float rectangle sample?
        aligned_boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    aligned_boxes = np.asarray(aligned_boxes, np.float32)
    labels = np.asarray(labels, np.int64)

    return index, encoded_image, aligned_boxes, labels, width, height

def parse_record(features):
    # tfrecord parser for TFRecordIterator (primitive data)

    index = int(features['index'][0])
    encoded_image = np.frombuffer(features['image'][0], dtype = np.uint8)
    width = int(features['width'][0])
    height = int(features['height'][0])
    boxes = features['boxes']

    assert len(boxes) % 5 == 0, 'Annotation error occured in box array.'
    box_cnt = len(boxes) // 5

    aligned_boxes = []
    labels = []

    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(boxes[i * 5]), float(boxes[i * 5 + 1]), float(boxes[i * 5 + 2]), float(boxes[i * 5 + 3]), float(boxes[i * 5 + 4])
        aligned_boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    aligned_boxes = np.asarray(aligned_boxes, np.float32)
    labels = np.asarray(labels, np.int64)

    return index, encoded_image, aligned_boxes, labels, width, height

def bbox_crop(bbox, crop_box=None, allow_outside_center=True):
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox

def bbox_iou(bbox_a, bbox_b, offset=0):
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)


def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def letterbox_resize(img, new_width, new_height, interp=0):
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def resize_with_bbox(img, bbox, new_width, new_height, interp=0, letterbox=False):
    if letterbox:
        image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox
    else:
        ori_height, ori_width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox


def random_flip(img, bbox, px=0, py=0):
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox


def random_expand(img, bbox, max_ratio=4, fill=0, keep_ratio=True):
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=img.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = img

    # correct bbox
    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox

def process_box(boxes, labels, img_size, class_num, anchors):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight. 
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
    # [N]
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print(feature_map_group, '|', y,x,k,c)

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(data, class_num, img_size, anchors, is_training, letterbox_resize):
    
    img_idx, encoded_img, boxes, labels, _, _ = parse_tfrecord(data)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

    ## I erased mix-up method here

    if is_training:
        # random color distortion
        img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 4)

        # random cropping
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=interp, letterbox=letterbox_resize)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1, letterbox=letterbox_resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(records, class_num, img_size, anchors, is_training, multi_scale=False, mix_up=False, letterbox_resize=True, interval=10):
    global iter_cnt

    # multi_scale training
    if multi_scale and is_training:
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # deleted mix up strategy
    
    for data in records:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(data, class_num, img_size, anchors, is_training, letterbox_resize)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch