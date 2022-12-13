# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QMainWindow, QHeaderView
from PyQt5.QtCore import QFile, QTimer, QDateTime, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtGui import QStandardItemModel, QImage, QPixmap
from PyQt5 import uic
import os
import sys
import psutil
import cv2 as cv
import numpy as np
import time


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
        self.btnStart.clicked.connect(self.StartWebCam)
        self.btnStop.clicked.connect(self.StopWebCam)

        tableModel = QStandardItemModel()
        header = ['Vehicle Type/Size', 'Vehicle Color',
                  'Vehicle Movement Direction', 'Vehicle Speed (km/h)']
        tableModel.setHorizontalHeaderLabels(header)
        self.table.setModel(tableModel)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.lcdTimer = QTimer()
        self.lcdTimer.timeout.connect(self.showTime)
        self.lcdTimer.start()

        self.resourceUsage = SystemMonitor()
        self.resourceUsage.start()
        self.resourceUsage.cpu.connect(self.getCpuUsage)
        self.resourceUsage.ram.connect(self.getRamUsage)

        self.rdbCamera.setChecked(True)
        self.btnSelect.setEnabled(False)
        self.pathVideoFile.setEnabled(False)

        self.rdbCamera.toggled.connect(lambda:self.buttonState(self.rdbCamera))
        self.rdbVideoFile.toggled.connect(lambda:self.buttonState(self.rdbVideoFile))

        self.actionQuit.triggered.connect(self.exit_program)
        self.btnClose.clicked.connect(self.exit_program)

    def showTime(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcdTime.display(self.DateTime.toString("hh:mm:ss"))

    def getCpuUsage(self, cpu):
        self.lblCPU.setText(str(cpu) + "%")
        if cpu > 15: self.lblCPU.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25: self.lblCPU.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45: self.lblCPU.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65: self.lblCPU.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85: self.lblCPU.setStyleSheet("color: rgb(237, 85, 59);")

    def getRamUsage(self, ram):
        self.lblRAM.setText(str(ram[2]) + "%")
        if ram[2] > 15: self.lblRAM.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25: self.lblRAM.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45: self.lblRAM.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65: self.lblRAM.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85: self.lblRAM.setStyleSheet("color: rgb(237, 85, 59);")

    def getFPS(self, fps):
        self.lblFPS.setText(str(fps))
        if fps > 5: self.lblFPS.setStyleSheet("color: rgb(237, 85, 59);")
        if fps > 15: self.lblFPS.setStyleSheet("color: rgb(60, 174, 155);")
        if fps > 25: self.lblFPS.setStyleSheet("color: rgb(85, 170, 255);")
        if fps > 35: self.lblFPS.setStyleSheet("color: rgb(23, 63, 95);")

    def buttonState(self, button):
        if button.text() == "Camera":
            if button.isChecked() == True:
                self.listCam.setEnabled(True)
                self.pathVideoFile.setEnabled(False)
                self.btnSelect.setEnabled(False)
            else:
                self.listCam.setEnabled(False)
                self.pathVideoFile.setEnabled(True)
                self.btnSelect.setEnabled(True)

        if button.text() == "Video File":
            if button.isChecked() == True:
                self.listCam.setEnabled(False)
                self.pathVideoFile.setEnabled(True)
                self.btnSelect.setEnabled(True)
            else:
                self.listCam.setEnabled(True)
                self.pathVideoFile.setEnabled(False)
                self.btnSelect.setEnabled(False)

    @pyqtSlot(np.ndarray)
    def opencv_emit(self, Image):
        original = self.cvt_cv_qt(Image)
        self.mainFrame.setPixmap(original)
        self.mainFrame.setScaledContents(True)

    def cvt_cv_qt(self, Image):
        offset = 5
        rgb_img = cv.cvtColor(src=Image, code=cv.COLOR_BGR2RGB)

        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        cvt2QtFormat = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(cvt2QtFormat)

        return pixmap  # QPixmap.fromImage(cvt2QtFormat)

    def StartWebCam(self):
        try:
            self.txtLog.append(
                f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Start Webcam ({self.listCam.currentText()})")
            self.btnStop.setEnabled(True)
            self.btnStart.setEnabled(False)

            global camIndex
            camIndex = self.listCam.currentIndex()

            # Opencv QThread
            self.worker = ThreadClass()
            self.worker.imageUpdate.connect(self.opencv_emit)
            self.worker.fps.connect(self.getFPS)
            self.worker.start()


        except Exception as error:
            pass

    def StopWebCam(self):
        self.txtLog.append(
            f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Stop Webcam ({self.listCam.currentText()})")
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        # Set Icon back to stop state
        self.worker.stop()

    def exit_program(self):
        self.ThreadActive = False
        sys.exit()

class SystemMonitor(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)

    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            self.cpu.emit(cpu)
            self.ram.emit(ram)
    def stop(self):
        self.ThreadActive = False
        self.quit()


class ThreadClass(QThread):
    imageUpdate = pyqtSignal(np.ndarray)
    fps = pyqtSignal(int)

    def run(self):
        global Capture
        capture = cv.VideoCapture(camIndex)

        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.ThreadActive = True
        prev_frame_time = 0
        while self.ThreadActive:
            ret, frame_cap = capture.read()
            new_frame_time = time.time()
            fpsValue = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            if ret:
                self.imageUpdate.emit(frame_cap)
                self.fps.emit(int(fpsValue))

    def stop(self):
        self.ThreadActive = False
        self.quit()
