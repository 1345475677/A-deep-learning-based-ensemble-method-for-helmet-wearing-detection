from PyQt5 import QtCore, QtGui, QtWidgets
from interface import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog
from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5.QtGui import QPixmap
import sys,os
from test_video import test_the_video


class helmet_detection(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(helmet_detection,self).__init__()
        self.setupUi(self)
        self.connector()
        self.show()

    def connector(self):
        self.queding.clicked.connect(self.press_queding)
        self.pathbutton.clicked.connect(self.press_pathbutton)
        self.myclose.clicked.connect(self.press_myclose)

    def press_pathbutton(self):
        file = QFileDialog()
        str = file.getOpenFileName()
        self.video_path = str[0]
        self.pathlabel.setText(self.video_path)

    def press_queding(self):
        self.video_path=self.pathlabel.text()
        if (self.video_path == ""):
            self.pathlabel.setText("there is no file")
            self.surelabel.setText("")
        elif(not os.path.exists(self.video_path)):
            self.pathlabel.setText("file path not exist")
            self.surelabel.setText("")
        else:
            self.surelabel.setText("running")
            t=int(self.time.text())
            test_the_video(self.video_path,t)
            self.surelabel.setText("finished")

    def press_myclose(self):
        self.close()
        exit()


def main():
    app=QApplication(sys.argv)
    my_helmet_detection=helmet_detection()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()