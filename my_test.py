# assumptions:
# 0. The "Environment Set-up" section has been completed using this guide - https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
# 1. the image has been taken so that the car is the right way up in the image
# 2. the car is not so far away that the umberplate cant be read
# 3. the car is not so close that the numberplate coveres the whole image
# 4. no classifiers can be used
# 5. no OCR can be used
# 6. SAM2 is being used with checkpoint sam2.1_hiera_base_plus.pt

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import os
import urllib
import numpy as np
import copy
import math
import scipy
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img     

sam2_checkpoint = "../checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

urls_string = "https://images.coches.com/_vn_/kia/Sportage/c399cf1d98a95d24f8e8715dd0b13fb2.jpg?p=cc_vn_high https://www.hyundai.com/es/zonaeco/image/catalog/articulos/ecodrive/electricos-2021/hyundai-nuevo-KONA-electrico-01.jpg https://www.topgear.com/sites/default/files/news-listicle/image/2022/11/p90419422_lowres_the-new-mini-3-door-_0.jpg?w=827&h=465 https://www.topgear.com/sites/default/files/2023/01/51952307109_b9b8d85293_k.jpeg?w=827&h=465 https://www.topgear.com/sites/default/files/2022/02/hyundai-uk-all-new-tucson-1220-50.jpeg?w=827&h=465 https://media.evo.co.uk/image/private/s--I_T1JLxX--/f_auto,t_content-image-full-desktop@1/v1696526185/evo/2023/10/BMW%20M2%20v%20Porsche%20Cayman%20GT4%20v%20Alpine%20A110%20R%20on%20road-11.jpg https://stage-drupal.car.co.uk/s3fs-public/styles/original_size/public/2019-09/why-are-number-plates-yellow-and-white.jpg?rt1UJUyIi7L2DpS613hFYlI5ng3U4QT3&itok=3SZjXU0B https://images.pistonheads.com/nimg/47298/blobid0.jpg https://mercedes-benz-club.co.uk/wp-content/uploads/elementor/thumbs/banner-image-1280x720-02-q1a5qhcmx0l65yryp4if6yzr8qhn55kpfwrneobd68.jpg https://static.dw.com/image/44096983_1004.webp https://media.euobserver.com/69f73b65af9807ac139dbd84a4473052.jpg"

image_urls = urls_string.split()

debug = False
show_images = False
save_images = True
show_overlays = True

min_w_h_ratio = 1.9
max_w_h_ratio = 5
min_plate_area = 1000
max_plate_area = 20000
min_occupied_percentage = 0.5
scale = 10
min_sub_ratio = 0.015
max_sub_ratio = 0.5
max_std_err = 0.22
max_center_dist = 40

stability_score_thresh = 0.8
mask_generator = SAM2AutomaticMaskGenerator(sam2, stability_score_thresh=stability_score_thresh)

image_no = 0
estimated_EU_cars = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for img_url in image_urls:
        image_no += 1
        EU = False
        filename = "image" + str(image_no)
        print("Processing", filename)
        image = img_from_url(img_url)
        display = copy.deepcopy(image)
        masks = mask_generator.generate(image)
        if debug:
            print(len(masks), " masks with keys", masks[0].keys())
        for i in range(len(masks)):
            sub_filename = filename + "_" + str(i) + ".png"
            f_x, f_y, f_w, f_h = masks[i]['bbox']
            x, y, w, h = int(f_x), int(f_y), int(f_w), int(f_h)
            seg_area = masks[i]['area']
            box_area = w * h
            ratio = w / h
            occupied = seg_area / box_area
            if debug:
                print("Seg ", i, " seg area: ", seg_area, " / box area: ", box_area," = occupied: ", occupied)
                print("    Box x, y, w, h: ", x, y, w, h, " ratio: ", ratio)
            color = (0, 0, 255)
            if min_plate_area < seg_area < max_plate_area:
                if debug:
                    print("    Found potential plate: ", min_plate_area, "<",  seg_area, "<", max_plate_area)
                if min_w_h_ratio < ratio < max_w_h_ratio and occupied > min_occupied_percentage:
                    color = (255, 0, 0)
                    potential_plate = copy.deepcopy(image[y:y+h, x:x+w])
                    display_plate = copy.deepcopy(potential_plate)
                
                    potential_plate_gray = cv2.cvtColor(potential_plate, cv2.COLOR_BGR2GRAY)
                    # ret2, potential_plate_bin = cv2.threshold(potential_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    min_pixel_value = np.min(potential_plate_gray)
                    max_pixel_value = np.max(potential_plate_gray)
                    pixel_range = max_pixel_value - min_pixel_value
                    if debug:
                        print("    potential plate min pixel value: ", min_pixel_value, " max pixel value: ", max_pixel_value, " pixel range: ", pixel_range)

                    if pixel_range < 175:
                        if debug:
                            print("    potential plate pixel range too small, skipping..........")
                        continue
                
                    sub_masks = mask_generator.generate(potential_plate)
                    no_sub_masks = len(sub_masks)
                    s_w, s_h = len(potential_plate[0]), len(potential_plate)
                    x_bound = int(s_w / 7)
                    y_bound = int(s_h / 6)
                    if debug:
                        print("    ", no_sub_masks, " sub masks - bounds: ", x_bound, y_bound, " seg area: ", seg_area)
                    if show_overlays and debug:
                        display_plate = cv2.rectangle(display_plate, (x_bound, y_bound), (len(potential_plate[0])-x_bound, len(potential_plate)-y_bound), (255, 0, 0), 2)
                    no_valid_sub_masks = 0
                    centers_x = np.array([])
                    centers_y = np.array([])
                    std_err = 1
                    blue = False    
                    for j in range(no_sub_masks):
                        sf_x, sf_y, sf_w, sf_h = sub_masks[j]['bbox']
                        sub_seg_area = sub_masks[j]['area']
                        sub_center_x, sub_center_y = int(sf_x + sf_w/2), int(sf_y + sf_h/2)
                        sub_color = (0, 0, 255)
                        if debug:
                            print("        Sub mask ", j, " centers: ", sub_center_x, sub_center_y, "sub seg area: ", sub_seg_area, ", ratio: ", sub_seg_area/seg_area)

                        avg_color = potential_plate[sub_center_y][sub_center_x]
                        hsv_img = cv2.cvtColor(np.array([[avg_color, avg_color], [avg_color, avg_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)
                        
                        if debug:
                            print("         ave color: ", avg_color, ", hsv:", hsv_img[0][0])
                            # if 115 < int(hsv_img[0][0][0]) < 130 and 126 < int(hsv_img[0][0][1]) and int(hsv_img[0][0][2]) > 50:
                            #     print("         its blue")
                            #     blue = True
                        if avg_color[0] > 110 and avg_color[1] < 150 and avg_color[2] < 60:
                            if debug:
                                print("         its blue 2")
                            blue = True
                        


                        if x_bound < sub_center_x < s_w  - x_bound and \
                        y_bound < sub_center_y < s_h - y_bound and min_sub_ratio < sub_seg_area/seg_area < max_sub_ratio:
                            if debug:
                                print("             selected")
                            no_valid_sub_masks += 1
                            centers_x = np.append(centers_x, sub_center_x)
                            centers_y = np.append(centers_y, sub_center_y)
                            sub_color = (0, 255, 0)
                        if show_overlays and debug:
                            display_plate = cv2.circle(display_plate, (sub_center_x, sub_center_y), 3, sub_color, 2) 
                            display_plate = cv2.rectangle(display_plate, (int(sf_x), int(sf_y)), (int(sf_x + sf_w), int(sf_y + sf_h)), (255, 0, 0), 2)
                    if no_valid_sub_masks > 3:
                        if blue:
                            EU = True
                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(centers_x, centers_y)
                        # distance = |ax0 + by0 + c| / sqrt(a^2 + b^2)
                        a = slope
                        b = -1
                        c = intercept
                        center_dist = abs(a * sub_center_x + b * sub_center_y + c) / np.sqrt(a**2 + b**2)
                        if debug:
                            print(f"    plate slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}, center_dist: {center_dist}")
                        if std_err < max_std_err and center_dist < max_center_dist:
                            if debug:
                                print("    Found ", no_valid_sub_masks, " valid sub masks")
                            color = (0, 255, 0)

                    if save_images and debug:
                        cv2.imwrite(sub_filename, display_plate)
                    if show_images and debug:
                        cv2.imshow('display_plate', display_plate)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            if show_overlays:
                if color[1] == 255:
                    display = cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                if debug:
                    display = cv2.putText(display, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    display = cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                

        if EU:
            estimated_EU_cars.append(True)
            if show_overlays:
                display = cv2.putText(display, "EU", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            estimated_EU_cars.append(False)


        if save_images:
            cv2.imwrite(filename + ".png", display)
        if show_images:                    
            cv2.imshow('THEKER image', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
print("Estimated EU cars: ", estimated_EU_cars)