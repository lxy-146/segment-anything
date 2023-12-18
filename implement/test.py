import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/media/ilab/lxy/MVI_grading_2023/data_content/MVI/01_M0_T1_4200001.jpg')
device = 'cuda'
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
sam.to(device)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def seg_all():
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)


    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.show() 

if __name__ == '__main__':
    seg_all()