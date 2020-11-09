import cv2
import numpy as np

file_name = 'C:\\Users\\LG\\Desktop\\work\\1_Trim.mp4'
min_confidence = 0.5

def detectAndDisplay(frame):
    # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 416,416 픽셀단위로 만듬

    # 데이터를 모델에 로드하는 작업
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []  # 정확도
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) # 카테고리중 probility 큰값을 나타내는 것
            confidence = scores[class_id]
            if confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4) # 박스 노이즈를 없앰 non maximum suppression (NMS threshold:0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]],confidences[i]*100)
            print(i, label)
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y-5), font, 2, color, 2)
    cv2.imshow("Image", frame)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



# Load Yolo
net = cv2.dnn.readNetFromDarknet("D:\\PycharmProjects\\assignment\\work\\darknet\\yolov3.cfg","D:\\PycharmProjects\\assignment\\work\\darknet\\yolov3.weights")
classes = []
with open("D:\\PycharmProjects\\assignment\\work\\darknet\\data\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # (yolo작동방식)
colors = np.random.uniform(0, 255, size=(len(classes), 3))


#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame) # 영상을 하나한 frame단위로 실행
    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break





# https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/