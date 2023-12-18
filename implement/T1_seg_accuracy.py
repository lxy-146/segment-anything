import cv2
import os
import json
import copy

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda'
# sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
# sam = sam_model_registry["vit_l"](checkpoint="./sam_vit_l_0b3195.pth")
sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
with open('./segmentation.json', 'r') as inf:
    segmentation_gt = json.load(inf)
save_file = 'images_IOU/vit_h_0.5'
if not os.path.exists(save_file):
    os.mkdir(save_file)


def show_anns(anns):
    # if len(anns) == 0:
    #     return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # print(anns[0]['bbox'])
    img = np.ones((anns['segmentation'].shape[0], anns['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    # for ann in sorted_anns:
    ann = anns
    m = ann['segmentation']
    # print(m.shape)
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask
    ax.imshow(img)

def get_bbox(segmentation):
    max_x = 0
    max_y = 0
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            if segmentation[i][j]:
                if max_x < i:
                    max_x = i
                if max_y < j:
                    max_y = j
    return max_x, max_y

def draw_bbox(img, bbox, c, w=10):
    cv2.line(img, (bbox[2], bbox[1]), (bbox[2], bbox[3]), c, w, 4)
    cv2.line(img, (bbox[0], bbox[1]), (bbox[2], bbox[1]), c, w, 4)
    cv2.line(img, (bbox[2], bbox[3]), (bbox[0], bbox[3]), c, w, 4)
    cv2.line(img, (bbox[0], bbox[3]), (bbox[0], bbox[1]), c, w, 4)
    return img

def get_seg_gt(img_name):
    seg = segmentation_gt[img_name]['regions'][0]['shape_attributes']
    bbox = list()
    bbox.append(min(seg['all_points_x']))
    bbox.append(min(seg['all_points_y']))
    bbox.append(max(seg['all_points_x']))
    bbox.append(max(seg['all_points_y']))
    return bbox

def display_mask(img, mask, bbox, gt_bbox, img_name, IOU):
    tmp_img = copy.deepcopy(img)
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(img)
    show_anns(mask)
    # plt.show()
    plt.subplot(122)
    tmp_img = draw_bbox(tmp_img, bbox, (0, 255, 0))
    tmp_img = draw_bbox(tmp_img, gt_bbox, (255, 0, 0))
    plt.imshow(tmp_img)
    plt.text(0, 1, str(IOU))
    plt.savefig(f"{save_file}/{img_name}",dpi=300) 
    plt.close()
    # plt.show()

class IOU_G():
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        
    def input_bbox(self, bbox1, bbox2):
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.inter_bbox = []
        self.inter_bbox.append(max(bbox1[0], bbox2[0]))
        self.inter_bbox.append(max(bbox1[1], bbox2[1]))
        self.inter_bbox.append(min(bbox1[2], bbox2[2]))
        self.inter_bbox.append(min(bbox1[3], bbox2[3]))
    
    def _get_area(self, bbox):
        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            return 0
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def get_IOU(self):
        area_inter = self._get_area(self.inter_bbox)
        area_1 = self._get_area(self.bbox1)
        area_2 = self._get_area(self.bbox2)
        if area_inter <= 0:
            IOU = 0
        else:
            IOU = area_inter / (area_1 + area_2 - area_inter)
        return IOU
    
    def is_same(self):
        v = self.get_IOU()
        return v > self.threshold

def mask2gt(masks, img_name, img):
    gt_bbox = get_seg_gt(img_name)
    iou_g = IOU_G(0.5)
    find_same = False
    best_IOU = 0
    for mask in masks:
        # print('gt bbox:', gt_bbox)
        bbox_pre = mask['bbox']
        bbox_pre[3], bbox_pre[2] = get_bbox(mask['segmentation'])
        # print('pre bbox:', bbox_pre)
        iou_g.input_bbox(gt_bbox, bbox_pre)
        tmp_IOU = iou_g.get_IOU()
        print('IOU:', tmp_IOU)
        if tmp_IOU > best_IOU:
            best_IOU = tmp_IOU
            display_mask(img, mask, bbox_pre, gt_bbox, img_name, best_IOU)
        if iou_g.is_same():
            print('SAME!!!!!!!')
            find_same = True
        # display_mask(img, mask, bbox_pre, gt_bbox, img_name)
    return find_same
        

def seg_imgs():
    img_folder = '/media/ilab/lxy/MVI_grading_2023/data_content/T1img'
    imgs = os.listdir(img_folder)
    total_num = len(imgs)
    acc_num = 0
    not_find_list = []
    for idx, img_name in enumerate(imgs):
        img = cv2.imread(os.path.join(img_folder, img_name))
        print(f'{img_name} processing...')
        masks = mask_generator.generate(img)
        findif = mask2gt(masks, img_name, img)
        if findif:
            acc_num += 1
        else:
            not_find_list.append(img_name)
        # plt.imshow(img)
        # show_anns(masks)
        # plt.show() 
        # bbox = masks[0]['bbox']
        # bbox[3], bbox[2] = get_bbox(masks[0]['segmentation'])
        # print(bbox)
        # img = draw_bbox(img, bbox)
        # # img = draw_bbox(img, [534, 685, 740, 878]   )
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()  
        print(f'==========================CURRENT ACC:{acc_num / (idx + 1)}==========================')  
    print(f'==========================FINAL ACC:{acc_num / total_num}==========================')
    print(not_find_list)

if __name__ == '__main__':
    seg_imgs()