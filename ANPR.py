import torch
from ultralytics import SAM
import cv2
import urllib
import numpy as np
import copy

def img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

sam_model = SAM("sam2.1_b.pt")
if torch.cuda.is_available():
    sam_model.to('cuda')
    print("SAM model loaded onto GPU.")
else:
    print("CUDA not available, using CPU.")
sam_model.info()

min_w_h_ratio = 2
max_w_h_ratio = 5
min_plate_area = 2000
max_plate_area = 20000
min_occupied_percentage = 0.55
debug = True
scale = 10

urls_string = "https://images.coches.com/_vn_/kia/Sportage/c399cf1d98a95d24f8e8715dd0b13fb2.jpg?p=cc_vn_high https://www.hyundai.com/es/zonaeco/image/catalog/articulos/ecodrive/electricos-2021/hyundai-nuevo-KONA-electrico-01.jpg https://www.topgear.com/sites/default/files/news-listicle/image/2022/11/p90419422_lowres_the-new-mini-3-door-_0.jpg?w=827&h=465 https://www.topgear.com/sites/default/files/2023/01/51952307109_b9b8d85293_k.jpeg?w=827&h=465 https://www.topgear.com/sites/default/files/2022/02/hyundai-uk-all-new-tucson-1220-50.jpeg?w=827&h=465 https://media.evo.co.uk/image/private/s--I_T1JLxX--/f_auto,t_content-image-full-desktop@1/v1696526185/evo/2023/10/BMW%20M2%20v%20Porsche%20Cayman%20GT4%20v%20Alpine%20A110%20R%20on%20road-11.jpg https://stage-drupal.car.co.uk/s3fs-public/styles/original_size/public/2019-09/why-are-number-plates-yellow-and-white.jpg?rt1UJUyIi7L2DpS613hFYlI5ng3U4QT3&itok=3SZjXU0B https://images.pistonheads.com/nimg/47298/blobid0.jpg https://mercedes-benz-club.co.uk/wp-content/uploads/elementor/thumbs/banner-image-1280x720-02-q1a5qhcmx0l65yryp4if6yzr8qhn55kpfwrneobd68.jpg https://static.dw.com/image/44096983_1004.webp https://media.euobserver.com/69f73b65af9807ac139dbd84a4473052.jpg"
image_urls = urls_string.split()
for img_url in image_urls:
    image = img_from_url(img_url)
    results = sam_model(image)

    boxes = results[0].boxes
    masks = results[0].masks
    no_seg = len(boxes.xywh)
    print("Number of boxes: ", no_seg)
    for i in range(no_seg):
        f_x, f_y, f_w, f_h = boxes.xywh[i]
        x, y, w, h = int(f_x - f_w/2), int(f_y - f_h/2), int(f_w), int(f_h)
        seg_area = int(torch.sum(masks[i].data[0]))
        area = w * h
        ratio = w / h
        occupied = seg_area / area
        # print("Seg ", i, " seg area: ", seg_area, " / area: ", area," = occupied: ", occupied)
        # print("    Box x, y, w, h: ", x, y, w, h, " ratio: ", ratio)
        color = (0, 0, 255)
        if min_plate_area < seg_area < max_plate_area:
            if min_w_h_ratio < ratio < max_w_h_ratio and occupied > min_occupied_percentage:
                color = (255, 0, 0)
                potential_plate = copy.deepcopy(image[y:y+h, x:x+w])
            
                potential_plate_gray = cv2.cvtColor(potential_plate, cv2.COLOR_BGR2GRAY)
                ret2, potential_plate_bin = cv2.threshold(potential_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # potential_plate = cv2.resize(potential_plate, (int(w * scale), int(h * scale)))
            
                # sub_results = sam_model(potential_plate)
                # sub_boxes = sub_results[0].boxes
                # no_sub_seg = len(sub_boxes.xywh)
                # print("Seg ", i, " - no. sub segs: ", no_sub_seg)
                # for j in range(no_sub_seg):
                #     sf_x, sf_y, sf_w, sf_h = sub_boxes.xywh[j]
                #     center_x, center_y = int(sf_x), int(sf_y)
                #     print("     Sub seg ", j, " - center_x: ", center_x, ", center_y: ", center_y)
                #     potential_plate = cv2.putText(potential_plate, str(j), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('potential_plate', potential_plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        image = cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        
    cv2.imshow('THEKER image 1', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()