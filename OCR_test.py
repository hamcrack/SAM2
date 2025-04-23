import cv2
import numpy as np
import urllib
import easyocr

def img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image

# This needs to run only once to load the model into memory
reader = easyocr.Reader(['en'])

urls_string = "https://images.coches.com/_vn_/kia/Sportage/c399cf1d98a95d24f8e8715dd0b13fb2.jpg?p=cc_vn_high https://www.hyundai.com/es/zonaeco/image/catalog/articulos/ecodrive/electricos-2021/hyundai-nuevo-KONA-electrico-01.jpg https://www.topgear.com/sites/default/files/news-listicle/image/2022/11/p90419422_lowres_the-new-mini-3-door-_0.jpg?w=827&h=465 https://www.topgear.com/sites/default/files/2023/01/51952307109_b9b8d85293_k.jpeg?w=827&h=465 https://www.topgear.com/sites/default/files/2022/02/hyundai-uk-all-new-tucson-1220-50.jpeg?w=827&h=465 https://media.evo.co.uk/image/private/s--I_T1JLxX--/f_auto,t_content-image-full-desktop@1/v1696526185/evo/2023/10/BMW%20M2%20v%20Porsche%20Cayman%20GT4%20v%20Alpine%20A110%20R%20on%20road-11.jpg https://stage-drupal.car.co.uk/s3fs-public/styles/original_size/public/2019-09/why-are-number-plates-yellow-and-white.jpg?rt1UJUyIi7L2DpS613hFYlI5ng3U4QT3&itok=3SZjXU0B https://images.pistonheads.com/nimg/47298/blobid0.jpg https://mercedes-benz-club.co.uk/wp-content/uploads/elementor/thumbs/banner-image-1280x720-02-q1a5qhcmx0l65yryp4if6yzr8qhn55kpfwrneobd68.jpg https://static.dw.com/image/44096983_1004.webp https://media.euobserver.com/69f73b65af9807ac139dbd84a4473052.jpg"
image_urls = urls_string.split()
for img_url in image_urls:
    img = img_from_url(img_url)
    # run OCR
    results = reader.readtext(img)

    for res in results:
        # bbox coordinates of the detected text
        xy = res[0]
        print(xy)
        # x, y, w, h = xy[0], xy[1], xy[2], xy[3]
        # text results and confidence of detection
        det, conf = res[1], res[2]
        print(det, conf)
        # show time :)
        # image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # image = cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()