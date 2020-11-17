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
frame_count = 0
min_confidence = 0.5
#min_directions = 10  # 최소 10번은 한 방향으로 가야 방향 인식

height = 0
width = 0

trackers = []
trackables = {}

file_name = 'D:\\AI_republic\\data\\vedio\\dynamite_Trim2.mp4'
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

num = 0 # 사람 수 세기

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
    objects,num = tracker.update(rects,num)

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
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0], centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    # num 표시
    text = "{}: {}".format("num",num)
    cv2.putText(frame, text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.255,0),2)

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
cv2.destroyAllWindows()