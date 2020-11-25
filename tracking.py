from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
'''
[centroid 알고리즘] 사용
새로 detection된 object가 들어왔을때 기존 tracking되고있던 object와의 구분
euclidean distance를 측정하여 제일 가까운 것으로 계속 tracking 한다 -> ID tracking 방법
'''
# detect된 object들 객체를 생성
class Trackable:
        def __init__(self, objectID, centroid):
                self.objectID = objectID
                self.centroids = [centroid]

# Tracking
class Tracker:
        def __init__(self, maxDisappeared=50): # 50번 끊기면 다른 사람으로 인식
                self.nextObjectID = 0           # maxDisappeared : detection을 놓치는 경우 예외상황들을 잡아줌
                self.objects = OrderedDict()
                self.disappeared = OrderedDict()
                self.maxDisappeared = maxDisappeared
                self.positions = []

        def clock(self, objectID):
                deregister_time = time.time() - self.exist_time
                # 시 분 초로 표현
                # print(deregister_time)
                deregister_time -= 70  # 50 frame : almost 70 sec -> 이 차이는 영상이 어떻게 생겼는지에 따라 다 다르므로 가게 test영상 받으면 정확히 맞춰줌
                hours, minutes, seconds = 0, 0, deregister_time
                if deregister_time // 60 > 1:  # 분 단위
                        minutes = deregister_time // 60
                        seconds = deregister_time % 60
                        hours = 0

                        deregister_time = deregister_time // 60
                        if deregister_time // 60 > 1:  # 시간 단위
                                hours = deregister_time // 60
                                deregister_time = hours
                                minutes = deregister_time //60
                                seconds = minutes%60
                print('DEREGISTER ID {} : {}시 {}분 {:.2f}초'.format(objectID, hours, minutes, seconds))

        def register(self, centroid):
                self.exist_time = time.time()
                self.objects[self.nextObjectID] = centroid
                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        def deregister(self, objectID):
                del self.objects[objectID]
                del self.disappeared[objectID]
                self.clock(objectID)

        def update(self, rects,num):
        # 새로 들어오면 register / tracking 할 필요 없어지면 deregister
                if len(rects) == 0:
                        for objectID in list(self.disappeared.keys()):
                                self.disappeared[objectID] += 1
                                # detection을 놓칠 수 있기 때문에 프래임 수 50번정도는 기다리고 삭제함
                                if self.disappeared[objectID] > self.maxDisappeared:
                                        self.deregister(objectID)
                                        num-=1
                        return self.objects

                inputCentroids = np.zeros((len(rects), 2), dtype="int") # detect된 물체 개수만큼 빈 2차원 배열 생성
                for (i, (startX, startY, endX, endY)) in enumerate(rects):
                        cX = int((startX + endX) / 2.0) # centroid값 계산
                        cY = int((startY + endY) / 2.0)
                        inputCentroids[i] = (cX, cY) # 사람의 중심점 좌표 저장

                if len(self.objects) == 0:
                        for i in range(0, len(inputCentroids)):
                                self.register(inputCentroids[i])
                                num += 1

                else: # tracking object를 계속 모니터링하면서 좌표값 등을 업데이트
                        objectIDs = list(self.objects.keys())
                        objectCentroids = list(self.objects.values())

                        D = dist.cdist(np.array(objectCentroids), inputCentroids)
                        # scipy의 compute distance 함수를 사용하여 distnace 계산

                        rows = D.min(axis=1).argsort() # 거리 내림차순 정렬
                        cols = D.argmin(axis=1)[rows] # 최소 거리 저장
                        # print('rows:',rows) # ex) [0,3,2,1]
                        # print('cols:',cols) #     [2,0,0,0]

                        usedRows = set()
                        usedCols = set()
                        for (row, col) in zip(rows, cols):
                                if row in usedRows or col in usedCols:
                                        continue

                                objectID = objectIDs[row]
                                self.objects[objectID] = inputCentroids[col]
                                self.disappeared[objectID] = 0
                                
                                usedRows.add(row)
                                usedCols.add(col)

                        # 짝을 안지은 것을 찾아냄
                        unusedRows = set(range(0, D.shape[0])).difference(usedRows) # 기존의 detection 없어지면 unusedRow에 들어감
                        unusedCols = set(range(0, D.shape[1])).difference(usedCols) # 새로 들어온 object
                        # D.shape[0] : object "tracking" 하는 것의 개수, D.shape[1] : 화면에서 "detection"하는 개수
                        
                        if D.shape[0] >= D.shape[1]: # object가 잘 관리되고 있다
                                for row in unusedRows: # detection이 끊긴게 있으면
                                        objectID = objectIDs[row]
                                        self.disappeared[objectID] += 1 # disappeared를 카운트
                                        if self.disappeared[objectID] > self.maxDisappeared:
                                                self.deregister(objectID)
                                                num -= 1
                        else: # tracking개수 < detection개수 : object가 새로 detect된 경우
                                for col in unusedCols: # 새로 detection된 object를 register
                                        self.register(inputCentroids[col])
                                        num += 1

                return self.objects,num # tracking하는 물체들에 대한 값과 사람 수를 리턴
