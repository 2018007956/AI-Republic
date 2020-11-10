import cv2
import numpy as np
import time

file_name = 'D:\\AI_republic\\data\\vedio\\0.mp4'
min_confidence = 0.5

face_model = 'D:\\AI_republic\\data\\ai_cv\\model\\res10_300x300_ssd_iter_140000.caffemodel' # 영상 처리에 많이 사용되는 딥러닝 프레임워크
face_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\deploy.prototxt.txt' # 모델이 어떻게 구성됐는지에 대한 메타 정보
age_model = 'D:\\AI_republic\\data\\ai_cv\\model\\age_net.caffemodel'
age_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\age_deploy.prototxt'
gender_model = 'D:\\AI_republic\\data\\ai_cv\\model\\gender_net.caffemodel'
gender_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\gender_deploy.prototxt'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

title_name = 'Age and Gender Recognition'
min_confidence = 0.5
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300) # 페이스 모델이 300x300 이므로

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)



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
    detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) # 카테고리중 probility 큰값을 나타내는 것
            confidence = scores[class_id]
            if class_id==0 and confidence > min_confidence:
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
    return indexes, boxes



# Load Yolo
net = cv2.dnn.readNetFromDarknet("D:\\PycharmProjects\\assignment\\AI Republic\\darknet\\yolov3.cfg","D:\\PycharmProjects\\assignment\\AI Republic\\darknet\\yolov3.weights")
classes = []
with open("D:\\PycharmProjects\\assignment\\AI Republic\\darknet\\data\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # (yolo작동방식)
colors = np.random.uniform(0, 255, size=(100, 3)) #100 가지 색상 사용

detected = False
frame_count=0
tracker = cv2.TrackerKCF_create()

elapsed_time = 0
trackers = cv2.MultiTracker_create()
#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if not ret:
        exit()# 잘못 읽거나 비디오 끝나면
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    start_time = time.time()
    frame_count +=1
    if detected:
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x,y,w,h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x+2,y+h),(0,255,0),2)
    else:
        indexes, boxes = detectAndDisplay(frame) ###### 여기서 계속 detect 안하고 첫 frame (사람들)만 detect 후 tracking 사용
        for i in range(len(boxes)):
            if i in indexes:
                trackers.add(tracker, frame, tuple(boxes[0]))
        detected = True

    cv2.imshow("Image", frame)
    frame_time = time.time() - start_time
    elapsed_time += frame_time
    #print(" Frame {} time {}".format(frame_count,frame_time)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

