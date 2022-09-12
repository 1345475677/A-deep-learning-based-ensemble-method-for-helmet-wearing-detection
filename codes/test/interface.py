# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(969, 767)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pathbutton = QtWidgets.QPushButton(self.centralwidget)
        self.pathbutton.setGeometry(QtCore.QRect(240, 100, 201, 28))
        self.pathbutton.setObjectName("pathbutton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 160, 72, 15))
        self.label.setObjectName("label")
        self.queding = QtWidgets.QPushButton(self.centralwidget)
        self.queding.setGeometry(QtCore.QRect(140, 320, 93, 28))
        self.queding.setObjectName("queding")
        self.myclose = QtWidgets.QPushButton(self.centralwidget)
        self.myclose.setGeometry(QtCore.QRect(490, 320, 93, 28))
        self.myclose.setObjectName("myclose")
        self.surelabel = QtWidgets.QLabel(self.centralwidget)
        self.surelabel.setGeometry(QtCore.QRect(180, 260, 651, 16))
        self.surelabel.setText("")
        self.surelabel.setObjectName("surelabel")
        self.pathlabel = QtWidgets.QLineEdit(self.centralwidget)
        self.pathlabel.setGeometry(QtCore.QRect(150, 150, 701, 31))
        self.pathlabel.setObjectName("pathlabel")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 210, 111, 31))
        self.label_2.setObjectName("label_2")
        self.time = QtWidgets.QLineEdit(self.centralwidget)
        self.time.setGeometry(QtCore.QRect(150, 210, 71, 31))
        self.time.setObjectName("time")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(230, 210, 81, 31))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 969, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pathbutton.setText(_translate("MainWindow", "select file"))
        self.label.setText(_translate("MainWindow", "file path"))
        self.queding.setText(_translate("MainWindow", "sure"))
        self.myclose.setText(_translate("MainWindow", "close"))
        self.label_2.setText(_translate("MainWindow", "Annotated refresh frequency"))
        self.time.setText(_translate("MainWindow", "3"))
        self.label_3.setText(_translate("MainWindow", "fps"))
