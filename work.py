<<<<<<< HEAD
from tracking import Tracker, Trackable
import cv2
import numpy as np
import time

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
# elapsed_time = 0
OUTPUT_SIZE = (300, 300) # 페이스 모델이 300x300 이므로

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)




frame_size = (416,416) # yolo의 두번째 사이즈
frame_count = 0
min_confidence = 0.5
#min_directions = 10  # 최소 10번은 한 방향으로 가야 방향 인식

height = 0
width = 0

trackers = []
trackables = {}

file_name = 'D:\\AI_republic\\data\\vedio\\dynamite_Trim.mp4'
output_name = 'D:\\AI_republic\\data\\vedio\\output.avi'

# Load Yolo
net = cv2.dnn.readNetFromDarknet("D:\\PycharmProjects\\assignment\\AI Republic\\darknet\\yolov3.cfg","D:\\PycharmProjects\\assignment\\AI Republic\\darknet\\yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize Tracker
tracker = Tracker()

# initialize the video writer
writer = None

def writeFrame(img):
    # use global variable, writer
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                                 (img.shape[1], img.shape[0]), True)

    if writer is not None: # writer가 한번 생성되면 쭉 동영상 만듬
        writer.write(img)


vs = cv2.VideoCapture(file_name)
# loop over the frames from the video stream
while True:
    ret, frame = vs.read()

    if frame is None:
        print('### No more frame ###')
        break


    # frame_count += 1

    (height, width) = frame.shape[:2]

    # construct a blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, frame_size, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    rects = []

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:] # 식별한 물체의 배열과 확률값
            class_id = np.argmax(scores) # 가장 확률이 높은 물체 지정
            confidence = scores[class_id]
            # Filter only 'person'
            if class_id == 0 and confidence > min_confidence:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            rects.append([x, y, x + w, y + h])
            label = '{:,.2%}'.format(confidences[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # use Tracker
    objects = tracker.update(rects)

    '''Object를 계속 기억하도록 만들어야함'''
    # loop over the trackable objects
    for (objectID, centroid) in objects.items():
        # check if a trackable object exists with the object ID
        trackable = trackables.get(objectID, None) # 감지되는 사람마다 trackables 객체 생성 - 트래킹 속성
        if trackable is None:
            trackable = Trackable(objectID, centroid)
        else: # - 동선과 머문 시간 측정
            xy = [c for c in trackable.centroids] # 무게중심의 x,y좌표 가져옴
            # -> DB에 저장 -> 동선추적가능

            trackable.centroids.append(centroid)

        # store the trackable object in our dictionary
        trackables[objectID] = trackable
        text = "ID {}-{:.2f}".format(objectID,trackables[objectID].exist_time)
        cv2.putText(frame, text, (centroid[0] -30, centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    writeFrame(frame) # 느리니까 output동영상 만듬

    # show the output frame
    cv2.imshow("Frame", frame)
    # frame_time = time.time() - start_time
    # print("Frame {} time {}".format(frame_count, frame_time))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

vs.release()
writer.release()
=======
from tracking import Tracker, Trackable
import cv2
import numpy as np

# face_model = 'D:\\AI_republic\\data\\ai_cv\\model\\res10_300x300_ssd_iter_140000.caffemodel' # 영상 처리에 많이 사용되는 딥러닝 프레임워크
# face_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\deploy.prototxt.txt' # 모델이 어떻게 구성됐는지에 대한 메타 정보
# age_model = 'D:\\AI_republic\\data\\ai_cv\\model\\age_net.caffemodel'
# age_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\age_deploy.prototxt'
# gender_model = 'D:\\AI_republic\\data\\ai_cv\\model\\gender_net.caffemodel'
# gender_prototxt = 'D:\\AI_republic\\data\\ai_cv\\model\\gender_deploy.prototxt'
#
# age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# gender_list = ['Male','Female']
#
# title_name = 'Age and Gender Recognition'
# min_confidence = 0.5
# recognition_count = 0
# # elapsed_time = 0
# OUTPUT_SIZE = (300, 300) # 페이스 모델이 300x300 이므로
#
# detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
# age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
# gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)




frame_size = (416,416) # yolo의 두번째 사이즈
min_confidence = 0.5

trackers = []
trackables = {}

file_name = 'D:\\AI_republic\\data\\vedio\\dynamite_Trim.mp4'
output_name = 'D:\\AI_republic\\data\\vedio\\output222.avi'

# Load Yolo
net = cv2.dnn.readNetFromDarknet("D:\\PycharmProjects\\assignment\\AI Republic\\yolov3.cfg","D:\\PycharmProjects\\assignment\\AI Republic\\yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# ['yolo_82', 'yolo_94', 'yolo_106'] 세 개의 output layer저장

writer = None
def writeFrame(img):
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                                 (img.shape[1], img.shape[0]), True)
    if writer is not None: # writer가 한번 생성되면 쭉 동영상 만듬
        writer.write(img)

num = 0 # 사람 수 세기
tracker = Tracker() # 객체 생성
vs = cv2.VideoCapture(file_name)
while True:
    ret, frame = vs.read() # 프레임을 하나씩 읽음

    if frame is None:
        print('### No more frame ###')
        break

    (height, width) = frame.shape[:2] # 박스 표시할떄 사용

    ## construct a blob for YOLO model 'yolo 사용할때'
    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, frame_size, (0, 0, 0), True, crop=False)
    net.setInput(blob) # 전처리된 blob 네트워크에 입력
    outs = net.forward(output_layers) # object detection한 내용 저장

    rects = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:] # 식별한 물체의 배열과 확률값
            class_id = np.argmax(scores) # 가장 확률이 높은 물체 지정
            confidence = scores[class_id]
            # Filter only 'person'
            if class_id == 0 and confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width) # detection[0][1][2][3]: yolo의 상대적인 값
                center_y = int(detection[1] * height) # 실제 값을 곱해서 절대적 값 만듬
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates 시작점
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h]) # 시작점과 w,h

                confidences.append(float(confidence)) # NMS function 사용하기 위해
    # 박스 중복 제외
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4) # 0.4:NMS계수
    for i in range(len(boxes)):
        if i in indexes: # 중복 제외된 대표 박스들만 인식
            x, y, w, h = boxes[i]
            rects.append([x, y, x + w, y + h]) # 대표 박스들만 저장-> 화면에 표현할 박스
            label = '{:,.2%}'.format(confidences[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    objects,num = tracker.update(rects,num) # 좌표와 사람 수를 프레임 마다 갱신

    '''Object를 계속 기억하도록 만들어야함'''
    for (objectID, centroid) in objects.items():
        # trackable 객체 생성 (objectID, centroid)
        trackable = trackables.get(objectID) # tracking 되고 있는 리스트에 없을 경우 None 리턴
        if trackable is None: # 새로 detection된 object
            trackable = Trackable(objectID, centroid)
        else: # - 동선과 머문 시간 측정
            xy = [c for c in trackable.centroids] # 중심좌표 가져옴
            # -> DB에 저장 -> 동선추적가능
            trackable.centroids.append(centroid)
            print('{}: {}'.format(objectID,xy))

        trackables[objectID] = trackable # tracking하는 object를 배열에 저장
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0], centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    text = "{}: {}".format("num",num)
    cv2.putText(frame, text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.255,0),2)

    writeFrame(frame) # 느리니까 output동영상 만듬

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break



vs.release()
writer.release()
>>>>>>> origin/master
cv2.destroyAllWindows()