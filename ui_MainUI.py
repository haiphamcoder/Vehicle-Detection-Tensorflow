# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/haiqqez/Workspace/Vehicle-Detection-Tensorflow/MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1202, 954)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_3.addItem(spacerItem)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupSysMonitor = QtWidgets.QGroupBox(self.centralwidget)
        self.groupSysMonitor.setObjectName("groupSysMonitor")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupSysMonitor)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lblRAM = QtWidgets.QLabel(self.groupSysMonitor)
        self.lblRAM.setMinimumSize(QtCore.QSize(60, 0))
        self.lblRAM.setAlignment(QtCore.Qt.AlignCenter)
        self.lblRAM.setObjectName("lblRAM")
        self.gridLayout_2.addWidget(self.lblRAM, 1, 2, 1, 1)
        self.lblCPU = QtWidgets.QLabel(self.groupSysMonitor)
        self.lblCPU.setMinimumSize(QtCore.QSize(60, 0))
        self.lblCPU.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCPU.setObjectName("lblCPU")
        self.gridLayout_2.addWidget(self.lblCPU, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupSysMonitor)
        self.label_5.setMinimumSize(QtCore.QSize(60, 0))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupSysMonitor)
        self.label_3.setMinimumSize(QtCore.QSize(20, 0))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupSysMonitor)
        self.label_2.setMinimumSize(QtCore.QSize(60, 0))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.lblFPS = QtWidgets.QLabel(self.groupSysMonitor)
        self.lblFPS.setMinimumSize(QtCore.QSize(60, 0))
        self.lblFPS.setAlignment(QtCore.Qt.AlignCenter)
        self.lblFPS.setObjectName("lblFPS")
        self.gridLayout_2.addWidget(self.lblFPS, 1, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupSysMonitor, 0, 3, 1, 1)
        self.groupSource = QtWidgets.QGroupBox(self.centralwidget)
        self.groupSource.setObjectName("groupSource")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupSource)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pathVideoFile = QtWidgets.QLineEdit(self.groupSource)
        self.pathVideoFile.setReadOnly(True)
        self.pathVideoFile.setObjectName("pathVideoFile")
        self.gridLayout.addWidget(self.pathVideoFile, 3, 1, 1, 1)
        self.rdbCamera = QtWidgets.QRadioButton(self.groupSource)
        self.rdbCamera.setObjectName("rdbCamera")
        self.gridLayout.addWidget(self.rdbCamera, 1, 0, 1, 1)
        self.rdbVideoFile = QtWidgets.QRadioButton(self.groupSource)
        self.rdbVideoFile.setObjectName("rdbVideoFile")
        self.gridLayout.addWidget(self.rdbVideoFile, 3, 0, 1, 1)
        self.listCam = QtWidgets.QComboBox(self.groupSource)
        self.listCam.setObjectName("listCam")
        self.gridLayout.addWidget(self.listCam, 1, 1, 1, 1)
        self.btnSelect = QtWidgets.QPushButton(self.groupSource)
        self.btnSelect.setObjectName("btnSelect")
        self.gridLayout.addWidget(self.btnSelect, 3, 2, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupSource, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem1, 0, 0, 1, 1)
        self.groupControl = QtWidgets.QGroupBox(self.centralwidget)
        self.groupControl.setObjectName("groupControl")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupControl)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.btnStart = QtWidgets.QPushButton(self.groupControl)
        self.btnStart.setObjectName("btnStart")
        self.gridLayout_6.addWidget(self.btnStart, 0, 0, 1, 1)
        self.btnPause = QtWidgets.QPushButton(self.groupControl)
        self.btnPause.setObjectName("btnPause")
        self.gridLayout_6.addWidget(self.btnPause, 0, 1, 1, 1)
        self.btnStop = QtWidgets.QPushButton(self.groupControl)
        self.btnStop.setObjectName("btnStop")
        self.gridLayout_6.addWidget(self.btnStop, 1, 0, 1, 1)
        self.btnReset = QtWidgets.QPushButton(self.groupControl)
        self.btnReset.setObjectName("btnReset")
        self.gridLayout_6.addWidget(self.btnReset, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupControl, 0, 2, 1, 1)
        self.groupRealTime = QtWidgets.QGroupBox(self.centralwidget)
        self.groupRealTime.setObjectName("groupRealTime")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupRealTime)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.lcdTime = QtWidgets.QLCDNumber(self.groupRealTime)
        self.lcdTime.setMinimumSize(QtCore.QSize(200, 0))
        self.lcdTime.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lcdTime.setAutoFillBackground(False)
        self.lcdTime.setDigitCount(8)
        self.lcdTime.setMode(QtWidgets.QLCDNumber.Dec)
        self.lcdTime.setObjectName("lcdTime")
        self.gridLayout_8.addWidget(self.lcdTime, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupRealTime, 0, 4, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem2, 0, 5, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_3)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout_3.addItem(spacerItem3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupMainFrame = QtWidgets.QGroupBox(self.centralwidget)
        self.groupMainFrame.setObjectName("groupMainFrame")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupMainFrame)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.mainFrame = QtWidgets.QLabel(self.groupMainFrame)
        self.mainFrame.setMinimumSize(QtCore.QSize(640, 480))
        self.mainFrame.setMaximumSize(QtCore.QSize(640, 480))
        self.mainFrame.setStyleSheet("background: white;")
        self.mainFrame.setAlignment(QtCore.Qt.AlignCenter)
        self.mainFrame.setObjectName("mainFrame")
        self.gridLayout_10.addWidget(self.mainFrame, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupMainFrame, 3, 5, 2, 1)
        self.groupModels = QtWidgets.QGroupBox(self.centralwidget)
        self.groupModels.setObjectName("groupModels")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupModels)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.rdbModelMobilenetV1 = QtWidgets.QRadioButton(self.groupModels)
        self.rdbModelMobilenetV1.setObjectName("rdbModelMobilenetV1")
        self.gridLayout_12.addWidget(self.rdbModelMobilenetV1, 0, 0, 1, 1)
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupModels)
        self.radioButton_4.setObjectName("radioButton_4")
        self.gridLayout_12.addWidget(self.radioButton_4, 1, 0, 1, 1)
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupModels)
        self.radioButton_5.setObjectName("radioButton_5")
        self.gridLayout_12.addWidget(self.radioButton_5, 2, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupModels, 3, 8, 1, 1)
        self.groupOption = QtWidgets.QGroupBox(self.centralwidget)
        self.groupOption.setObjectName("groupOption")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupOption)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupOption)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout_4.addWidget(self.checkBox_2)
        self.checkBox = QtWidgets.QCheckBox(self.groupOption)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_4.addWidget(self.checkBox)
        self.gridLayout_4.addWidget(self.groupOption, 3, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem4, 3, 9, 1, 1)
        self.groupSetting = QtWidgets.QGroupBox(self.centralwidget)
        self.groupSetting.setObjectName("groupSetting")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.groupSetting)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.groupBox = QtWidgets.QGroupBox(self.groupSetting)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_14.addWidget(self.label_7, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_14.addWidget(self.label, 0, 0, 1, 1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_2.setObjectName("spinBox_2")
        self.gridLayout_14.addWidget(self.spinBox_2, 0, 3, 1, 1)
        self.spinBox_4 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_4.setObjectName("spinBox_4")
        self.gridLayout_14.addWidget(self.spinBox_4, 1, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_14.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_14.addWidget(self.label_4, 0, 2, 1, 1)
        self.spinBox_3 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_3.setObjectName("spinBox_3")
        self.gridLayout_14.addWidget(self.spinBox_3, 1, 1, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_14.addWidget(self.spinBox, 0, 1, 1, 1)
        self.checkBox_3 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_3.setText("")
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_14.addWidget(self.checkBox_3, 2, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setMinimumSize(QtCore.QSize(80, 0))
        self.label_8.setObjectName("label_8")
        self.gridLayout_14.addWidget(self.label_8, 2, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_14.addWidget(self.pushButton, 2, 3, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_14, 0, 0, 1, 1)
        self.gridLayout_13.addWidget(self.groupBox, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupSetting, 4, 3, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem5, 3, 6, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.txtLog = QtWidgets.QTextEdit(self.groupBox_8)
        self.txtLog.setReadOnly(True)
        self.txtLog.setObjectName("txtLog")
        self.gridLayout_11.addWidget(self.txtLog, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_8, 4, 8, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem6, 3, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem7, 3, 4, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_4)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.groupBox_9 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.table = QtWidgets.QTableView(self.groupBox_9)
        self.table.setObjectName("table")
        self.gridLayout_9.addWidget(self.table, 0, 0, 1, 1)
        self.horizontalLayout_4.addWidget(self.groupBox_9)
        spacerItem9 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem9)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_3.addLayout(self.verticalLayout_6)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem10)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem11)
        self.btnClose = QtWidgets.QPushButton(self.centralwidget)
        self.btnClose.setObjectName("btnClose")
        self.horizontalLayout.addWidget(self.btnClose)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem12)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1202, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Vehicle Detection"))
        self.groupSysMonitor.setTitle(_translate("MainWindow", "System Monitor"))
        self.lblRAM.setText(_translate("MainWindow", "0"))
        self.lblCPU.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "RAM"))
        self.label_3.setText(_translate("MainWindow", "CPU"))
        self.label_2.setText(_translate("MainWindow", "FPS"))
        self.lblFPS.setText(_translate("MainWindow", "0"))
        self.groupSource.setTitle(_translate("MainWindow", "Source"))
        self.rdbCamera.setText(_translate("MainWindow", "Camera"))
        self.rdbVideoFile.setText(_translate("MainWindow", "Video File"))
        self.btnSelect.setText(_translate("MainWindow", "Select"))
        self.groupControl.setTitle(_translate("MainWindow", "Control"))
        self.btnStart.setText(_translate("MainWindow", "Start"))
        self.btnPause.setText(_translate("MainWindow", "Pause"))
        self.btnStop.setText(_translate("MainWindow", "Stop"))
        self.btnReset.setText(_translate("MainWindow", "Reset"))
        self.groupRealTime.setTitle(_translate("MainWindow", "Real Time"))
        self.groupMainFrame.setTitle(_translate("MainWindow", "Main"))
        self.mainFrame.setText(_translate("MainWindow", "Main Source"))
        self.groupModels.setTitle(_translate("MainWindow", "Models"))
        self.rdbModelMobilenetV1.setText(_translate("MainWindow", "SSD Mobilenet V1 Coco"))
        self.radioButton_4.setText(_translate("MainWindow", "SSD2"))
        self.radioButton_5.setText(_translate("MainWindow", "SSD3"))
        self.groupOption.setTitle(_translate("MainWindow", "Option"))
        self.checkBox_2.setText(_translate("MainWindow", "Export CSV file"))
        self.checkBox.setText(_translate("MainWindow", "Export output video"))
        self.groupSetting.setTitle(_translate("MainWindow", "Setting"))
        self.groupBox.setTitle(_translate("MainWindow", "ROI line"))
        self.label_7.setText(_translate("MainWindow", "H: "))
        self.label.setText(_translate("MainWindow", "X: "))
        self.label_6.setText(_translate("MainWindow", "W: "))
        self.label_4.setText(_translate("MainWindow", "Y: "))
        self.label_8.setText(_translate("MainWindow", "Draw ROI"))
        self.pushButton.setText(_translate("MainWindow", "SET"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Log"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Traffic Measurement"))
        self.btnClose.setText(_translate("MainWindow", "Close"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
