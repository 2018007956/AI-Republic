from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time


class Trackable: # tracking되는 객체를 생성
        def __init__(self, objectID, centroid):
                self.objectID = objectID
                self.centroids = [centroid]
                self.counted = False

class Tracker:
        def __init__(self, maxDisappeared=50): # 50번 끊기면 다른 사람으로 인식
                self.nextObjectID = 0
                self.objects = OrderedDict()
                self.disappeared = OrderedDict()
                self.maxDisappeared = maxDisappeared
                self.positions = []

        def clock(self, objectID):
                deregister_time = time.time() - self.exist_time
                # 시 분 초로 표현
                # print(deregister_time)
                deregister_time -= 40  # 50 frame : almost 70 sec -> 이 차이는 영상이 어떻게 생겼는지에 따라 다 다르므로 가게 test영상 받으면 정확히 맞춰줌
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
                # print('------------',time.time())

        def register(self, centroid):
                # print('REGISTER ID {} register time: {}'.format(self.nextObjectID,time.time()))
                self.exist_time = time.time()
                self.objects[self.nextObjectID] = centroid
                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        def deregister(self, objectID):
                del self.objects[objectID]
                del self.disappeared[objectID]
                # print('DEREGISTERED : ' + str(objectID))
                self.clock(objectID)

        def update(self, rects,num):
                if len(rects) == 0:
                        for objectID in list(self.disappeared.keys()):
                                self.disappeared[objectID] += 1
                                # detection을 놓칠 수 있기 때문에 프래임 수 50번정도는 기다리고 삭제함
                                if self.disappeared[objectID] > self.maxDisappeared:
                                        self.deregister(objectID)
                                        num-=1
                        return self.objects

                inputCentroids = np.zeros((len(rects), 2), dtype="int")

                for (i, (startX, startY, endX, endY)) in enumerate(rects):
                        cX = int((startX + endX) / 2.0)
                        cY = int((startY + endY) / 2.0)
                        inputCentroids[i] = (cX, cY)

                if len(self.objects) == 0:
                        for i in range(0, len(inputCentroids)):
                                self.register(inputCentroids[i])
                                num += 1
                else:
                        objectIDs = list(self.objects.keys())
                        objectCentroids = list(self.objects.values())

                        D = dist.cdist(np.array(objectCentroids), inputCentroids)

                        rows = D.min(axis=1).argsort()

                        cols = D.argmin(axis=1)[rows]

                        usedRows = set()
                        usedCols = set()

                        for (row, col) in zip(rows, cols):
                                if row in usedRows or col in usedCols:
                                        continue

                                objectID = objectIDs[row]
                                self.objects[objectID] = inputCentroids[col]
                                self.disappeared[objectID] = 0
                                
                                usedRows.add(row) # tracking계속 되는데 detection이 끊긴것
                                usedCols.add(col) # 새로운 사람 들어옴

                        unusedRows = set(range(0, D.shape[0])).difference(usedRows) # detection 없어지면 unusedRow에 들어감
                        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                        
                        if D.shape[0] >= D.shape[1]:
                                for row in unusedRows:
                                        objectID = objectIDs[row]
                                        self.disappeared[objectID] += 1

                                        if self.disappeared[objectID] > self.maxDisappeared:
                                                self.deregister(objectID)
                                                num -= 1
                        else:
                                for col in unusedCols:
                                        self.register(inputCentroids[col])
                                        num += 1

                return self.objects,num
