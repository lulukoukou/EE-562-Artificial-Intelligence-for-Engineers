##NAME:QIHAN ZHAO
## ID: 1726473

# !/usr/bin/python2.7
import math

def astar(filename):
    # totalObstacle stores all the rectangulars
    # totalChoice stores all the points of rectangulars
    totalObstacle = []
    totalChoice = []
    initialState = (0, 0)
    goldState = (0, 0)
    # read the input data
    # intital state, gold state, num of obstacles, all the rectangulars
    with open(filename) as file:
        cnt = 1
        for line in file.readlines():
            tmpList = line.split()
            if cnt == 1:
                initialState = (int(tmpList[0]), int(tmpList[1]))
                cnt += 1
            elif cnt == 2:
                goldState = (int(tmpList[0]), int(tmpList[1]))
                cnt += 1
            elif cnt == 3:
                numObstacles = int(tmpList[0])
                cnt += 1
            else:
                obstacle = []
                i = 0
                while i < len(tmpList):
                    obstacle.append((int(tmpList[i]), int(tmpList[i + 1])))
                    i += 2
                totalObstacle.append(obstacle)
                totalChoice = totalChoice + obstacle
                #totalObstacle = totalObstacle + obstacle


    # function intersection is to check whether two lines are intersected, if intersects then return True, otherwises return False
    # 1. project to the x and y axis to make sure the point is not inside the rectangular
    # 2. inner multiplication to make sure the point is on the same side
    def intersection(point1, point2, point3, point4):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]

        x3 = point3[0]
        y3 = point3[1]
        x4 = point4[0]
        y4 = point4[1]
        if not (min(x1, x2) < max(x3, x4) and min(y3, y4) < max(y1, y2) and min(x3, x4) < max(x1, x2) and min(y1, y2) < max(
                y3, y4)):
            return False

        fc = (y3 - y1) * (x2 - x1) - (x3 - x1) * (y2 - y1)
        fd = (y4 - y1) * (x2 - x1) - (x4 - x1) * (y2 - y1)

        if fc * fd > 0:
            return False
        return True


    # function interrectangular is to check whether the selected line is intersected with any rectangulars
    def interrectangular(point_check, point_choose, rectangular):
        #print (rectangular)
        check1 = intersection(point_check, point_choose, rectangular[0], rectangular[2])
        check2 = intersection(point_check, point_choose, rectangular[1], rectangular[3])
        return check1 or check2

    # function calculate is to calculate the straight distance from point1 to point2
    def calculate(point1, point2):
        return math.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)

    # openList stores all the generated and not explored points
    # closeList sotres all the explored points
    # each node includes: value g, value h, value f = g + h, coordinate location, parent location
    key_loc = "point"
    key_g = "g"
    key_h = "h"
    key_f = "f"
    key_successor = "successor"
    key_parent = "parent"
    openList = []
    g = 0
    h = calculate(initialState, goldState)
    sum = g + h
    openList.append({key_g: g, key_h: h, key_f: h, key_loc: initialState, key_successor: [], key_parent: initialState})
    closeList = []
    # flag is to show whether the selected line has intersected with any rectangulars
    flag = True

    # function inopenlist is to check whether the generated node has already been stored in openList
    def inopenlist(point):
        tmp = [p[key_loc] for p in openList]
        if point in tmp:
            return True
        return False


    # function incloselist is to check whether the generated node has already been moved to closeList
    def incloselist(point):
        tmp = [p[key_loc] for p in closeList]
        if point in tmp:
            return True
        return False


    # function getindex is to get the index of the given coordinate location
    def getindex(point, list):
        tmp = [p[key_loc] for p in list]
        return tmp.index(point)


    # if openList becomes empty, then fails
    # otherwises, it will break until finding out the goldState
    while openList:
        # openList is sorted according to the value g in ascending order
        # the first one in openList will be always with the smallest value g
        tmpDict = openList[0]
        del openList[0]
        # move this node to closeList to indicate that this node has been explored
        closeList.append(tmpDict)
        parentState = tmpDict[key_loc]
        if parentState == goldState:
            break
        g = tmpDict[key_g]

        point_check = (parentState[0], parentState[1])
        # select point from totalChoice list
        # test whether the line (generated by the point_check and point_choose) has intersected with any rectangular
        for point_choose in totalChoice:
            if calculate(point_check, point_choose) == 0:
                continue

            for item in totalObstacle:
                if interrectangular(point_check, point_choose, item):
                    flag = False

            # if flag == True, it indicates that the selected has not intersected with any rectangular
            # the point_choose will be added to the successor list of the point_check
            # the value g needs to be update

            # condition1: if the point_choose has exited in openList, we need to update the value g and it's parent
            # if the current value g is smaller than the original one

            # condition2: if the point_choose has exited in closeList, we need to update the value g and it's parent
            # if the current value g is smaller than the original one, then move this node to openList

            # condition3: if the point_choose has never exited before, we need to append the new node to openList
            # then to choose next point and repeat checking

            if flag:
                tmpDict[key_successor].append(point_choose)
                g_choose = g + calculate(point_check, point_choose) # update g
                h = calculate(point_choose, goldState)
                sum = g_choose + h

                if inopenlist(point_choose):
                    index = getindex(point_choose, openList)
                    g_original = openList[index].get(key_g)
                    if g_original > g_choose:
                        openList[index][key_g] = g_choose
                        openList[index][key_f] = g_choose + openList[index][key_h]
                        openList[index][key_parent] = parentState

                elif incloselist(point_choose):
                    index = getindex(point_choose, closeList)
                    g_original = closeList[index].get(key_g)
                    if g_original > g_choose:
                        closeList[index][key_g] = g_choose
                        closeList[index][key_f] = g_choose + closeList[index][key_h]
                        closeList[index][key_parent] = parentState
                        openList.append(closeList[index])
                        del closeList[index]
                else:
                    openList.append({key_g: g_choose, key_h: h, key_f: sum, key_loc: point_choose, key_successor: [], key_parent: parentState})

            flag = True
        openList.sort(key=lambda x: x[key_f]) # sort openList

    # result stores the optimal path
    # result_g stores value g for the optimal paht
    result = []
    result_g = []
    curState = goldState
    result.append(curState)
    result_g.append(0)

    while curState != initialState:
        index = getindex(curState, closeList)
        tmp_dict = closeList[index]
        parent_state = tmp_dict[key_parent]
        curState = parent_state
        cur_g = tmp_dict[key_g]
        result.append(curState)
        result_g.append(cur_g)

    # sort both result and result_g in ascending order
    result.sort()
    result_g.sort()

    # print result out as required format
    print ("Point\t\tCumulative Cost\n")
    for i in range(len(result)):
        print (str(result[i]) + "\t" + str(round(result_g[i], 8)) + "\n")


print ("solution for simple dataset\n")
filename1 = "./data1.txt"
astar(filename1)

print ("solution for difficult dataset\n")
filename2 = "./data2.txt"
astar(filename2)

print ("solution for self-designed dataset\n")
filename3 = "./data3.txt"
astar(filename3)