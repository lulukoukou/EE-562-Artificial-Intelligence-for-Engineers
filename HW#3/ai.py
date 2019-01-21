import time
import random 
import io


class key:
    def key(self):
        return "10jifn2eonvgp1o2ornfdlf-1230"

class ai:
    def __init__(self):
        pass

    class State:
        def __init__(self, a, b, a_fin, b_fin):
            self.a = a
            self.b = b
            self.a_fin = a_fin
            self.b_fin = b_fin
            

    # Kalah:
    #         b[5]  b[4]  b[3]  b[2]  b[1]  b[0]
    # b_fin                                         a_fin
    #         a[0]  a[1]  a[2]  a[3]  a[4]  a[5]
    # Main function call:
    # Input:
    # a: a[5] array storing the stones in your holes
    # b: b[5] array storing the stones in opponent's holes
    # a_fin: Your scoring hole (Kalah)
    # b_fin: Opponent's scoring hole (Kalah)
    # t: search time limit (ms)
    # a always moves first
    #
    # Return:
    # You should return a value 0-5 number indicating your move, with search time limitation given as parameter
    # If you are eligible for a second move, just neglect. The framework will call this function again
    # You need to design your heuristics.
    # You must use minimax search with alpha-beta pruning as the basic algorithm
    # use timer to limit search, for example:
    # start = time.time()
    # end = time.time()
    # elapsed_time = end - start
    # if elapsed_time * 1000 >= t:
    #    return result immediately 
    def move(self, a, b, a_fin, b_fin, t):
        curState = self.State(a, b, a_fin, b_fin)
        # To test the execution time, use time and file modules
        # In your experiments, you can try different depth, for example:
        f = open('time.txt', 'a') #append to time.txt so that you can see running time for all moves.
        # Make sure to clean the file before each of your experiment
        for d in [8]: #You should try more
            f.write('depth = '+str(d)+'\n')
            t_start = time.time()
            #self.minimax(depth = d)
            action = self.minimax(d, curState, t + t_start)
            f.write(str(time.time()-t_start)+'\n')
        f.close()
        return action
        #But remember in your final version you should choose only one depth according to your CPU speed (TA's is 3.4GHz)
        # My CPU Speed: 2.5GHz
        #and remove timing code. 
        
        #Comment all the code above and start your code here

    # calling function
    # minmax is the basic seraching algorithm for the AI
    # 1. use maxValue to get the best next step value
    # 2. use successors_AI to generate the list including next state associated with the value
    # 3. return the index of the hole, which value is the same as previously generated one
    # 4. next move will move all the stones in that hole
    # 5. the minimax alogrithm will return the best next step based on the current position and predicting opponent's potential steps
    def minimax(self, depth, curState, finalT):
        time.sleep(0.1*depth)
        v, s = self.maxValue(curState, float('-inf'), float('inf'), depth, finalT)
        a, state_list, _ = self.successors_AI(curState, finalT)
        index = 0
        
        for i in range(len(state_list)):
            tmpS = [s.b_fin] + s.a + [s.a_fin] + s.b
            if tmpS == state_list[i]:
                index = i
        
        #print (a[index])
        if a[index] < 7:
            return a[index] - 1
        else:
            return 14 - a[index]

    # maxValue return the choice for AI
    # maxValue will be chosen among the list containing opponent's choices
    # alpha prunning is due to that we only require the maxValue
    # any values that are smaller than the current maxValue could be cut off
    def maxValue(self, state, alpha, beta, depth, finalT):
        if self.terminal_test(state, depth, finalT):
            return self.utility(state)
        v = float('-inf')
        _, state_list, _ = self.successors_AI(state, finalT)
        #print (state_list)
        for s in state_list:
            s = self.State(s[1:7], s[8:14], s[7], s[0])
            v = max(v, self.minValue(s, alpha, beta, depth - 1, finalT))
            if v >= beta:
                return v, s
            alpha = max(alpha, v)
        return v, s
    
    # minValue return the potential choices for opponent
    # beta prunnint is due to that we only require the minValue
    # any values that are larger than the current minValue could be cut off
    def minValue(self, state, alpha, beta, depth, finalT):
        if self.terminal_test(state, depth, finalT):
            return self.utility(state)
        v = float('inf')
        _, state_list, _ = self.successors_OP(state, finalT)
        for s in state_list:
            s = self.State(s[1:7], s[8:14], s[7], s[0])
            v = min(v, self.maxValue(s, alpha, beta, depth - 1, finalT))
            if v <= beta:
                return v, s
            beta = min(beta, v)
        return v, s

    # terminal_test determines when to exit the minValue and maxValue searching process
    # based on the game rules, if it reaches the top of the tree
    # or if one of side has no more stones
    # or if time out
    # the system should automatically exit
    # otherwise, alpha-beta prunning searching will coutinue until one of above conditions is met
    def terminal_test(self, state, depth, finalT):
        if (depth == 0):
            return True

        if (sum(state.a)) == 0 or (sum(state.b) == 0): 
            return True

        if (time.time() > finalT):
            return True
        
        return False
    
    # successors_AI generates the successors for the AI side
    # I transform the state object into the list c, which is more convinent for calling and using
    # we need to test each hole and record the potential next step for each choice
    # for example,
    # for the first hole, we know how many stones inside and the game should follow the rules
    # that distribute those stones into following hoels except opponent's kalah, at the same time
    # extra turn or tall all the other side's stones might happen
    # this process will be iterated over each hole
    # only successor for each choice will be generated for successor_AI, the utility function will be designed to evaluate 
    def successors_AI(self, state, finalT):
        # b_fin: 0, a_fin: 7
        # list c - a: 1 2 3 4 5 6
        #             0 1 2 3 4 5
        # list c - b: 8 9 10 11 12 13
        #             0 1  2  3  4  5


        suc = []
        state_list = []
        c = [state.b_fin] + state.a + [state.a_fin] + state.b
        #print (c)
        #print (state.a)
        # if you run out of stones on your side, the opponent takes all the stones left on his side and plus them in his kalah
        if (sum(state.b) == 0):
            state.a_fin += sum(state.a)
            state.a = state.b
            c = [state.b_fin] + state.a + [state.a_fin] + state.b
            suc  [0]
            state_list.append(c)
            return suc, state_list, c

        for index in range(1, len(state.a) + 1):
            numStone = c[index]
            # if this hole is empty, skip it
            if numStone == 0:
                continue
            
            # get the successor and whether there is extra turn based on game rules
            newC, newTurn = self.rule(c, index, 1, numStone, True)
            #print (newTurn)
            newState = self.State(newC[1:7], newC[8:14], newC[7], newC[0])
            tmpSuc = index
            suc.append(tmpSuc)
            state_list.append(newC)

            if self.terminal_test(newState, 1, finalT):
                return suc, state_list, newC

            # if there is the extra turn, recall successor_AI
            # remember to store the newly generated successors associated with thier index
            if newTurn:
                prevsuc, state_list, prevC = self.successors_AI(newState, finalT)
                suc = suc + prevsuc
                state_list.append(prevC)
        
        # suc includes the index for all the generated successors
        # state_list includes all the generated successors
        # c is the previousState, which is previous to the newState
        return suc, state_list, c

    
    # successors_OP generates the successors for the OP side
    # before the AI makes the decision, it should predict which step the opponent will take
    # that is why in minValue we will use successors_OP to generate successor and choose beta based on this function
    # I transform the state object into the list c, which is more convinent for calling and using
    # we need to test each hole and record the potential next step for each choice
    # for example,
    # for the first hole, we know how many stones inside and the game should follow the rules
    # that distribute those stones into following hoels except AI's kalah, at the same time
    # extra turn or tall all the other side's stones might happen
    # this process will be iterated over each hole
    # only successor for each choice will be generated for successor_OP, the utility function will be designed to evaluate 

    def successors_OP(self, state, finalT):
        # b_fin: 0, a_fin: 7
        # list c - a: 1 2 3 4 5 6
        #             0 1 2 3 4 5
        # list c - b: 8 9 10 11 12 13
        #             0 1  2  3  4  5


        suc = []
        state_list = []
        c = [state.b_fin] + state.a + [state.a_fin] + state.b
        #print (c)
        #print (state.a)
        # if you run out of stones on your side, the opponent takes all the stones left on his side and plus them in his kalah
        if (sum(state.a) == 0):
            state.b_fin += sum(state.b)
            state.b = state.a
            c = [state.b_fin] + state.a + [state.a_fin] + state.b
            suc = [0]
            state_list.append(c)
            return suc, state_list, c

        for index in range(8, len(state.b) + 8):
            numStone = c[index]
            if numStone == 0:
                continue
            # get the successor and whether there is extra turn based on game rules
            newC, newTurn = self.rule(c, index, 1, numStone, False)
            #print (newTurn)
            newState = self.State(newC[1:7], newC[8:14], newC[7], newC[0])
            tmpSuc = index
            suc.append(tmpSuc)
            state_list.append(newC)

            if self.terminal_test(newState, 1, finalT):
                return suc, state_list, newC

            if newTurn:
                prevsuc, state_list, prevC = self.successors_OP(newState, finalT)
                suc = suc + prevsuc
                state_list.append(prevC)
            
        # suc includes the index for all the generated successors
        # state_list includes all the generated successors
        # c is the previousState, which is previous to the newState
        return suc, state_list, newC
                

    # there are several rules that should be followed
    # 1. if the last stone lands in its own kalah, an extra turn
    # 2. if the last stone lands in its own empty hone, take all the stones in the opponent's opposite hole and put them in your kalah
    def rule(self, c, index, count, numStone, isMe):
        
        newTurn = False
        
        for i in range(1, numStone + 1):
            cur_index = index + count
            if c[cur_index % 14] == 0:
                continue
            # isMe - is AI
            # if the cur_index is not the opponent's kalah
            # we need to test whether there is last stone landing in AI's empty hole or AI's kalah
            # newTurn determines whether there is an extra turn
            # at last, each following hole adds one stone until all stones are distributed
            if not isMe:
                if cur_index % 14 != 0:
                    if cur_index % 14 >= 1 and cur_index % 14 <= 6 and c[cur_index % 14] == 0 and i == numStone:
                        c[cur_index % 14] += c[14 - (cur_index % 14)]
                        c[14 - (cur_index % 14)] = 0
                
                    elif cur_index % 14 == 7 and i == numStone:
                        newTurn = True
                
                else:
                    cur_index += 1
                    count += 1
            
                c[cur_index % 14] += 1
                count += 1  
            # if the cur_index is not the opponent's kalah
            # we need to test whether there is last stone landing in OP's empty hole or OP's kalah
            # newTurn determines whether there is an extra turn
            # at last, each following hole adds one stone until all stones are distributed
            else:
                if cur_index % 14 != 7:
                    if cur_index % 14 >= 8 and cur_index % 14 <= 13 and c[cur_index % 14] == 0 and i == numStone:
                        c[cur_index % 14] += c[14 - (cur_index % 14)]
                        c[14 - (cur_index % 14)] = 0
                
                    elif cur_index % 14 == 0 and i == numStone:
                        newTurn = True
                else:
                    cur_index += 1
                    count += 1
                
                c[cur_index % 14] += 1
                count += 1  
        # return the generated state and the flag for whether there is a new turn or not
        return c, newTurn
    # heuristic function is used to evaluate each step
    # detailed explanation on the report
    # brief description:
    # the heuristic function is designed based on : 
    # difference between (a_fin, b_fin), whether there is extra turn, empty holes, much less balls than the opponents
    def utility(self, state):
        returnValue = 0

        if state.a_fin > state.b_fin:
            returnValue += 0.5 * (state.a_fin + state.b_fin)
        
        else:
            returnValue += 0.5 * (state.a_fin + state.b_fin)
        
        if sum(state.a) > sum(state.b):
            returnValue += 0.3 * (sum(state.a) - sum(state.b))

        else:
            returnValue += 0.3 * (sum(state.a) - sum(state.b))
        
        for item in state.a:
            if item == 0:
                returnValue -= 1
        
        for item in state.b:
            if item == 0:
                returnValue += 1

        return returnValue
        

