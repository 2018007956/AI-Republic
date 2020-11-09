import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

image_path = 'C:\\Users\\LG\\Desktop\\tensor\\3.jpg' # 여기에는 테스트할 이미지의 경로 및 이름을 넣어주시면 됩니다.
im = cv2.imread(image_path) # 이미지 읽기


# object detection (물체 검출)
bbox, label, conf = cv.detect_common_objects(im)

# print(bbox, label, conf)
if len(bbox)==0:
    print('검출된 것이 없습니다')
else:
    im = draw_bbox(im, bbox, label, conf)
    cv2.imshow('img',im)
    cv2.waitKey()
#cv2.imwrite('C:\\Users\\LG\\Desktop\\tensor\\result.jpg', im)


# https://bskyvision.com/678