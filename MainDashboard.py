# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QMainWindow, QHeaderView
from PyQt5.QtCore import QFile
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5 import uic
import os
import sys


class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_ui()

    def load_ui(self):
        path = os.path.join(os.path.dirname(__file__), "MainUI.ui")
        uiFile = QFile(path)
        uiFile.open(QFile.ReadOnly)
        self.ui = uic.loadUi(uiFile, self)
        uiFile.close()

        self.onlineCam = QCameraInfo.availableCameras()
        self.listCam.addItems([c.description() for c in self.onlineCam])

        tableModel = QStandardItemModel()
        tableModel.setHorizontalHeaderLabels(['Vehicle Type/Size', 'Vehicle Color', 'Vehicle Movement Direction', 'Vehicle Speed (km/h)'])
        for i in range(10):
            tableModel.setItem(i, QStandardItem("Car"))
        self.table.setModel(tableModel)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.actionQuit.triggered.connect(self.exit_program)
        self.btnClose.clicked.connect(self.exit_program)


    def exit_program(self):
        sys.exit()
