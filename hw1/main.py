# Name: QIHAN ZHAO
# ID: 1726473
# UW EE 562 HW#1

## state records # of missionary on the left side, # of cabbinal on the left side, side of boat
class State:
    def __init__(self, numofmissionary, numofcabbinal, side):
        self.missionary = numofmissionary
        self.cabbinal = numofcabbinal
        self.side = side

## define the initial state and final state as required
initialState = State(3, 3, "L")
finalState = State(0, 0, "R")

## the action list include how the number of missionary and the number of cabbinal will change while boat moving from left to right or from right to left
## "L" indicates that boat moves from Left to Right, "R" indicates that boat moves from Right to Left
action = [[-1, 0, "L"], [-2, 0, "L"], [0, -1, "L"], [0, -2, "L"], [-1, -1, "L"], [1, 0, "R"], [2, 0, "R"], [0, 1, "R"],
          [0, 2, "R"], [1, 1, "R"]]

## stack store the path when recursive
stack = []
stack.append(initialState)

## result will store all the possible path
result = []
opp = "R"
## contIll = # of illegal states, contRep = # of repeated states, contTotal = # of all states exclude illegal and repeated
contIll = 0
contRep = 0
contTotal = 0


## check function is used to make sure # of missionary and # of cabbinal on each side is in range(0, 3), and outnumber does not happen
## if outnumber happens, it will be considered as illegal state
def check(missionary, cabbinal):
    global contIll
    if missionary >= 0 and missionary <= 3 and cabbinal >= 0 and cabbinal <= 3:
        if missionary >= cabbinal or missionary == 0:
            return True
        else:
            contIll += 1
    return False


## printout, printstate, printlist are all used to print out the state object or to test intermediate states
def printout(allstate):
    print("solution:")
    for item in allstate:
        print("(" + str(item.missionary) + "," + str(item.cabbinal) + "," + item.side + ")")
def printstate(item):
    print(str(item.missionary) + "," + str(item.cabbinal) + "," + item.side)
def printlist(l):
    for item in l:
        printout(item)

## eaual function is used to test whether the finalstate is reached or not
def equal(state1, state2):
    result = True
    if state1.missionary == state2.missionary and state1.cabbinal == state2.cabbinal and state1.side == state2.side:
        result = True
    else:
        result = False
    return result


## checkinlist is to check whether the state object has been added in to the stack
## if the state object already exists, it will be counted as repeat
def checkinlist(item, l):
    for ll in l:
        if equal(item, ll):
            return True
    return False


## Recursive Depth-first Search
## the function will return when stack is empty
## stack will be added into the result if the finalstate is reached
## when current state is reached, next state is generated based on the action list
## if next state satisfy all the requirements (no outnumber on each side), it will be appended into stack
## if next state has been tested and finished, stack will pop out last state and search for next state
def dfs(curstate):
    global contTotal
    global contRep

    if (len(stack) == 0):
        return

    if (equal(curstate, finalState)):
        result.append(stack.copy())
        stack.pop()
        return

    curstate = stack.pop()
    contTotal += 1
    for nextAction in action:
        if (nextAction[2] == curstate.side):
            m = curstate.missionary + nextAction[0]
            c = curstate.cabbinal + nextAction[1]

            if curstate.side == "R":
                opp = "L"
            else:
                opp = "R"

            nextState = State(m, c, opp)  # left side
            nextOppState = State(3 - m, 3 - c, curstate.side)

            if check(nextState.missionary, nextState.cabbinal) and check(nextOppState.missionary,
                                                                         nextOppState.cabbinal):

                if len(stack) == 0 or not checkinlist(nextState, stack):
                    stack.append(curstate)
                    stack.append(nextState)
                    dfs(nextState)
                    stack.pop()
                else:
                    if len(stack) != 0:
                        contRep += 1;



dfs(initialState)
printlist(result)
print ("totals: " + str(contTotal) + " " + "Illegals: " + str(contIll) + " " + "Repeats: " + str(contRep) + " ")

