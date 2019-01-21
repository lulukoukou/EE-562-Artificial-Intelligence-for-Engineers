import sys
import threading
import time
import os
import uuid
import suds
import random

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from ui import *


class startUI(QMainWindow):
    def __init__(self):
        QWidget.__init__(self)
        self.allui = Ui_Dialog()
        self.allui.setupUi(self)
        self.allui.wMain.hide()
        self.resize(420, 226)
        self.setMaximumWidth(420)
        self.setMinimumWidth(420)
        self.playing = False
        global gametype, t
        global aifile, hostguid
        global guids, names
        global aimodule, url, modulekey, client
        global state
        global a, b, a_fin, b_fin
        global firstMove
        firstMove = False
        t = 1000
        a = [6, 6, 6, 6, 6, 6]
        b = [6, 6, 6, 6, 6, 6]
        a_fin = 0
        b_fin = 0
        state = self.strState(False)
        url = 'http://craniowin2.cs.washington.edu/kalah/port.asmx?wsdl'
        modulekey = "10jifn2eonvgp1o2ornfdlf-1230"
        client = suds.client.Client(url)
        self.connect(self.allui.pbPlayHuman, SIGNAL("clicked()"), self.playHuman)
        self.connect(self.allui.pbSelectFile, SIGNAL("clicked()"), self.selectFile)
        self.connect(self.allui.pbCreate, SIGNAL("clicked()"), self.createHost)
        self.connect(self.allui.pbCancel, SIGNAL("clicked()"), self.cancelCreate)
        self.connect(self.allui.pbGo, SIGNAL("clicked()"), self.playInternet)
        self.tH = threading.Thread(target=self.updateHostList)
        self.tH.setDaemon(True)
        self.tH.start()
        self.tC = threading.Thread()
        self.tM = threading.Thread()
        self.connect(self.allui.pbA1, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbA2, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbA3, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbA4, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbA5, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbA6, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB1, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB2, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB3, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB4, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB5, SIGNAL("clicked()"), self.PBclicked)
        self.connect(self.allui.pbB6, SIGNAL("clicked()"), self.PBclicked)
        self.updateControl()

    def setButtons(self, flag):
        self.allui.pbA1.setEnabled(False)
        self.allui.pbA2.setEnabled(False)
        self.allui.pbA3.setEnabled(False)
        self.allui.pbA4.setEnabled(False)
        self.allui.pbA5.setEnabled(False)
        self.allui.pbA6.setEnabled(False)
        if a[0] > 0:
            self.allui.pbA1.setEnabled(flag)
        if a[1] > 0:
            self.allui.pbA2.setEnabled(flag)
        if a[2] > 0:
            self.allui.pbA3.setEnabled(flag)
        if a[3] > 0:
            self.allui.pbA4.setEnabled(flag)
        if a[4] > 0:
            self.allui.pbA5.setEnabled(flag)
        if a[5] > 0:
            self.allui.pbA6.setEnabled(flag)
        self.allui.pbB1.setEnabled(False)
        self.allui.pbB2.setEnabled(False)
        self.allui.pbB3.setEnabled(False)
        self.allui.pbB4.setEnabled(False)
        self.allui.pbB5.setEnabled(False)
        self.allui.pbB6.setEnabled(False)
        self.allui.pbA.setEnabled(False)
        self.allui.pbB.setEnabled(False)
        self.update()

    def closeEvent(self, event):
        global hostguid
        if 'hostguid' in globals().keys():
            client.service.removeHost(hostguid)
            client.service.updateState(hostguid, str(-1))
        if self.tC.is_alive():
            self.tC.join(0)
        if self.tH.is_alive():
            self.tH.join(0)
        if self.tM.is_alive():
            self.tM.join(0)
        event.accept()

    def openFile(self):
        global aimodule
        try:
            aimodule = __import__('ai')
            key = getattr(aimodule, 'key')
        except:
            return 1
        m = key()
        check = False
        for fun in dir(m):
            if fun == 'key':
                check = True
        if not check or (check and m.key() != modulekey):
            return 1
        else:
            return 0

    def selectFile(self):
        flag = self.openFile()
        if flag == 1:
            self.allui.lbFile.setText('File error!')
        else:
            self.allui.lbFile.setText('AI File opened')
            global aifile
            aifile = os.path.abspath('./ai.py')

    def cancelCreate(self):
        #self.allui.lbStatus.setText('Canceled')
        self.allui.lvHosts.setEnabled(True)
        self.allui.pbCreate.setEnabled(True)
        self.allui.pbGo.setEnabled(True)
        self.allui.cbInternetOption.setEnabled(True)
        self.allui.tbHostName.setEnabled(True)
        self.allui.pbCancel.setEnabled(False)
        client.service.removeHost(hostguid)

    def createHost(self):
        global gametype, hostguid
        idx = self.allui.cbInternetOption.currentIndex()
        if idx == 1:
            rtn = self.openFile()
            if rtn == 1:
                QMessageBox.information(self, '', 'AI File open error')
                return
            else:
                gametype = 3
        else:
            gametype = 2
        hostguid = str(uuid.uuid1())
        hostname = self.allui.tbHostName.toPlainText()
        ret = client.service.registerHost(hostguid, hostname, gametype)
        self.tC = threading.Thread(target=self.checkMatch, args=())
        self.tC.setDaemon(True)
        self.tC.start()

    def changeDialogSize(self):
        if not self.playing:
            self.setMaximumWidth(542)
            self.setMinimumWidth(542)
            self.resize(542, 226)
            self.allui.wMain.show()
            self.allui.wStart.hide()
        self.playing = not self.playing

    def updateHostList(self):
        global guids, names
        guids = []
        names = []
        while True:
            curid = -1
            curguid = ""
            flag = False
            if self.allui.lvHosts.currentRow() >= 0:
                curid = self.allui.lvHosts.currentRow()
                curguid = guids[curid]
                flag = True

            self.allui.lvHosts.clear()
            guids = []
            names = []
            hosts = []
            webstring = client.service.queryHost()
            if webstring != None:
                hosts = webstring.split('$$')
            for item in hosts:
                if len(item) > 0:
                    var = item.split('##')
                    guids.append(var[0])
                    names.append(var[1])
                    self.allui.lvHosts.addItem(var[1])
            if flag and curguid in guids:
                curid = guids.index(curguid)
                if curid >= 0:
                    self.allui.lvHosts.setCurrentRow(curid)
            time.sleep(8)

    def playHuman(self):
        global gametype, t, aifile
        if 'aifile' not in globals().keys():
            QMessageBox.information(self, '', 'Open AI file first!')
            return
        gametype = 1
        t = int(self.allui.cbTime.currentText())
        self.changeDialogSize()
        self.allui.wStart.setEnabled(False)
        self.startA()
        if self.tC.is_alive():
            self.tC.join(1)
        if self.tH.is_alive():
            self.tH.join(1)

    def playInternet(self):
        global gametype, hostguid, curguid
        idx = self.allui.lvHosts.currentRow()
        if idx >= 0:
            hostguid = guids[idx]
            rtn = client.service.checkMatched(hostguid)
            if rtn == -1:
                QMessageBox.information(self, '', 'Host closed')
                return
            curguid = str(uuid.uuid1())
            client.service.matchHost(hostguid, curguid)
            rtn = client.service.checkMatched(hostguid)
            if rtn == 2:
                gametype = 2
            elif rtn == 3:
                if self.openFile() == 1:
                    QMessageBox.information(self, '', 'AI File open error')
                    return
                gametype = 3
            else:
                QMessageBox.information(self, '', 'Bad status!')
            client.service.createState(hostguid, curguid, state)
            self.changeDialogSize()
            self.allui.wStart.setEnabled(False)
            self.startB()
            if self.tC.is_alive():
                self.tC.join(1)
            if self.tH.is_alive():
                self.tH.join(1)
        else:
            QMessageBox.information(self, '', 'Select a game!')

    def checkMatch(self):
        self.allui.pbCancel.setEnabled(True)
        self.allui.lvHosts.setEnabled(False)
        self.allui.pbCreate.setEnabled(False)
        self.allui.pbGo.setEnabled(False)
        self.allui.cbInternetOption.setEnabled(False)
        self.allui.tbHostName.setEnabled(False)
        while True:
            rtn = client.service.checkMatched(hostguid)
            if rtn != 1 and gametype == rtn:
                self.changeDialogSize()
                self.allui.wStart.resize(0, 0)
                self.allui.wStart.setEnabled(False)
                self.startA()
                if self.tH.is_alive():
                    self.tH.join(1)
                return
            time.sleep(3)

    def checkWin(self):
        if a_fin > 36:
            self.allui.lStatus.setText('Game Over. You win!')
            return 3
        if b_fin > 36:
            self.allui.lStatus.setText('Game Over. You lose!')
            return 1
        if a_fin == 36 and b_fin == 36:
            self.allui.lStatus.setText('Game Over. Draw!')
            return 2
        return 0

    def strState(self, swap):
        if swap:
            self.swap()
        s = str(a[0]) + ',' + str(a[1]) + ',' + str(a[2]) + ',' + str(a[3]) + ',' + str(a[4]) + ',' + str(a[5]) + '#' \
               + str(a_fin) + '#' + str(b[0]) + ',' + str(b[1]) + ',' + str(b[2]) + ',' + str(b[3]) + ',' + str(b[4]) + ',' + \
               str(b[5]) + '#' + str(b_fin)
        if swap:
            self.swap()
        return s

    def swap(self):
        global a, b, a_fin, b_fin
        c=a[:]
        a=b[:]
        b=c[:]
        c = a_fin
        a_fin = b_fin
        b_fin = c


    def updateLocalState(self, move):
        global a, b, a_fin, b_fin
        ao = a[:]
        all = a[move:] + [a_fin] + b + a[:move]
        count = a[move]
        all[0] = 0
        p = 1
        while count > 0:
            all[p] += 1
            p = (p + 1) % 13
            count -= 1
        a_fin = all[6 - move]
        b = all[7 - move:13 - move]
        a = all[13 - move:] + all[:6-move]
        cagain = bool()
        ceat = False
        p = (p - 1) % 13
        if p == 6 - move:
            cagain = True
        if p <= 5 - move and ao[move] < 14:
            id = p + move
            if (ao[id] == 0 or p % 13 == 0) and b[5 - id] > 0:
                ceat = True
        elif p >= 14 - move and ao[move] < 14:
            id = p + move - 13
            if (ao[id] == 0 or p % 13 == 0) and b[5 - id] > 0:
                ceat = True
        if ceat:
            a_fin += a[id] + b[5-id]
            b[5-id] = 0
            a[id] = 0
        if sum(a)==0:
            b_fin += sum(b)
        if sum(b)==0:
            a_fin += sum(a)

        return cagain, ceat

    def updateControl(self):
        self.allui.pbA.setText(str(a_fin))
        self.allui.pbB.setText(str(b_fin))
        self.allui.pbA1.setText(str(a[0]))
        self.allui.pbA2.setText(str(a[1]))
        self.allui.pbA3.setText(str(a[2]))
        self.allui.pbA4.setText(str(a[3]))
        self.allui.pbA5.setText(str(a[4]))
        self.allui.pbA6.setText(str(a[5]))
        self.allui.pbB1.setText(str(b[0]))
        self.allui.pbB2.setText(str(b[1]))
        self.allui.pbB3.setText(str(b[2]))
        self.allui.pbB4.setText(str(b[3]))
        self.allui.pbB5.setText(str(b[4]))
        self.allui.pbB6.setText(str(b[5]))
        self.update()


    def startA(self):
        if gametype == 1:
            self.setButtons(True)
            self.allui.lStatus.setText("Playing against local AI. Your turn.")
        elif gametype == 2:
            self.setButtons(True)
            self.allui.lStatus.setText("Playing via internet. Your turn.")
        elif gametype == 3:
            global a,b,a_fin,b_fin,t,state, firstMove
            firstMove = True
            self.setButtons(False)
            self.allui.lStatus.setText("Playing against internet AI.")
            entry_ai = getattr(aimodule, 'ai')
            ai = entry_ai()
            move = ai.move(a[:], b[:], a_fin, b_fin, t)
            if a[move] == 0 :
                print "illegal move, quit!"
                sys.exit()
                
            cagain, ceat = self.updateLocalState(move)

            state = self.strState(False)
            print "A" + str(move) + " " + state + " " + str(ceat)
            self.updateControl()
            cwin = self.checkWin()
            if cwin!=0:
                    return
            while cagain:
                self.allui.lStatus.setText("Bingo! Move again!")
                move = ai.move(a[:], b[:], a_fin, b_fin, t)
                if a[move] == 0 :
                    print "illegal move, quit!"
                    sys.exit()
                
                cagain, ceat = self.updateLocalState(move)

                state = self.strState(False)
                print "A" + str(move) + " " + state + " " + str(ceat)
                self.updateControl()
                cwin = self.checkWin()
                if cwin!=0:
                    return
            rtn = client.service.updateState(hostguid, self.strState(False))
            if rtn == 1:
                self.allui.lStatus.setText("Error")
            self.setButtons(False)
            self.tM = threading.Thread(target=self.moveB)
            self.tM.setDaemon(True)
            self.tM.start()

    def startB(self):
        if gametype == 2 or gametype == 3:
            self.allui.lStatus.setText("Playing via internet. Waiting for opponent.")
            self.setButtons(False)
            self.tM = threading.Thread(target=self.moveB)
            self.tM.setDaemon(True)
            self.tM.start()

    def PBclicked(self):
        global state

        sender = self.sender()
        self.setButtons(False)
        move = int(sender.objectName()[3])-1
        cagain, ceat = self.updateLocalState(move)
        print "A" + str(move)+" "+self.strState(False) + " " + str(ceat)
        self.updateControl()
        state = self.strState(False)
        cwin = self.checkWin()
        if cwin!=0:
            if gametype == 2 or gametype == 3:
                client.service.updateState(hostguid, state)
            return
        if cagain:
            self.allui.lStatus.setText("Bingo! Move again!")
            self.setButtons(True)
            return

        if gametype == 2:
            rtn = client.service.updateState(hostguid, state)
            if rtn == 1:
                self.allui.lStatus.setText("Error")
        self.tM = threading.Thread(target=self.moveB)
        self.tM.setDaemon(True)
        self.tM.start()

    def updateRemoteState(self, status, flag):
        global state, a, b, a_fin, b_fin
        ary = status.split('#')
        if flag:
            ary_b = ary[2].split(',')
            ary_a = ary[0].split(',')
            a_fin = int(ary[1])
            b_fin = int(ary[3])
        else:
            ary_b = ary[0].split(',')
            ary_a = ary[2].split(',')
            a_fin = int(ary[3])
            b_fin = int(ary[1])
        for i in range(6):
            a[i] = int(ary_a[i])
            b[i] = int(ary_b[i])

    def moveB(self):
        global state, a, b, a_fin, b_fin
        if gametype == 1:
            self.allui.lStatus.setText("Playing against local AI. Waiting for opponent.")
            entry_ai = getattr(aimodule, 'ai')
            ai = entry_ai()
            self.swap()

            move = ai.move(a[:], b[:], a_fin, b_fin, t)
            
            if a[move] == 0 :
                print "illegal move (from player B), quit!"
                sys.exit()
                
            cagain, ceat = self.updateLocalState(move)
            print "B" + str(move) + " " + self.strState(True) + " " + str(ceat)
            self.swap()
            self.updateControl()
            cwin = self.checkWin()
            if cwin!=0:
                    return
            while cagain:
                self.allui.lStatus.setText("Bingo! Move again!")
                self.swap()
                move = ai.move(a[:], b[:], a_fin, b_fin, t)
                if a[move] == 0 :
                    print "illegal move (from player B), quit!"
                    sys.exit()
                cagain, ceat = self.updateLocalState(move)
                print "B" + str(move) + " " + self.strState(True) + " " + str(ceat)
                self.swap()
                self.updateControl()
                cwin = self.checkWin()
                if cwin!=0:
                    return
            self.setButtons(True)
            self.allui.lStatus.setText("Opponent has moved " + str(move+1) + ". Your turn.")
        elif gametype == 2:
            self.allui.lStatus.setText('Waiting for opponent move.')
            status = client.service.getState(hostguid)
            while not status or status == state:
                status = client.service.getState(hostguid)
                time.sleep(2)
            if status == str(-1):
                self.allui.lStatus.setText('Your opponent quit, you win!')
                self.setButtons(False)
                return
            self.updateRemoteState(status, False)
            self.updateControl()
            print "B" + " " + self.strState(True)
            cwin = self.checkWin()
            if cwin!=0:
                return
            self.setButtons(True)
            self.allui.lStatus.setText("Opponent has moved. Your turn.")
        elif gametype == 3:
            while True:
                status = client.service.getState(hostguid)
                while not status or status == state:
                    status = client.service.getState(hostguid)
                    time.sleep(2)
                if status == str(-1):
                    self.allui.lStatus.setText('Your opponent quit, you win!')
                    self.setButtons(False)
                    return
                self.updateRemoteState(status, firstMove)
                state = self.strState(not firstMove)
                self.updateControl()
                print "B " + " " + state
                cwin = self.checkWin()
                if cwin!=0:
                    return

                entry_ai = getattr(aimodule, 'ai')
                ai = entry_ai()
                move = ai.move(a[:], b[:], a_fin, b_fin, t)
                cagain, ceat = self.updateLocalState(move)
                state = self.strState(not firstMove)
                print "A" + str(move) + " " + state
                self.updateControl()
                cwin = self.checkWin()
                if cwin!=0:
                    client.service.updateState(hostguid, state)
                    return
                while cagain:
                    self.allui.lStatus.setText("Bingo! Move again!")
                    move = ai.move(a[:], b[:], a_fin, b_fin, t)
                    cagain, ceat = self.updateLocalState(move)
                    state = self.strState(not firstMove)
                    print "A" + str(move) + " " + state
                    self.updateControl()
                    cwin = self.checkWin()
                    if cwin!=0:
                        client.service.updateState(hostguid, state)
                        return

                state = self.strState(not firstMove)
                rtn = client.service.updateState(hostguid, state)
                self.allui.lStatus.setText("Playing against internet AI.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    aui = startUI()
    aui.show()
    sys.exit(app.exec_())
