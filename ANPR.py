import torch
from ultralytics import SAM
import cv2
import urllib
import numpy as np

def img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img

sam_model = SAM("sam2.1_b.pt")
if torch.cuda.is_available():
    sam_model.to('cuda')
    print("SAM model loaded onto GPU.")
else:
    print("CUDA not available, using CPU.")
sam_model.info()

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
        occupied = seg_area / area
        print("Seg ", i, " seg area: ", seg_area, " / area: ", area," = occupied: ", occupied)
        print("    Box x, y, w, h: ", x, y, w, h)
        color = (0, 0, 255)
        if 2000 < seg_area < 20000:
            if 1 < w / h < 5 and occupied > 0.6:
                color = (0, 255, 0)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        image = cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        
    cv2.imshow('THEKER image 1', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()