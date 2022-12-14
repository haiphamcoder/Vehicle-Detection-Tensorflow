from PyQt5.QtWidgets import QMainWindow, QHeaderView, QFileDialog
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
import csv
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util


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
        self.btnStart.clicked.connect(self.StartDetection)
        self.btnStop.clicked.connect(self.StopDetection)
        self.rdbModelMobilenetV1.setChecked(True)

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
        self.rdbCamera.toggled.connect(
            lambda: self.buttonState(self.rdbCamera))
        self.rdbVideoFile.toggled.connect(
            lambda: self.buttonState(self.rdbVideoFile))
        self.btnSelect.clicked.connect(self.openFileNameDialog)

        self.actionQuit.triggered.connect(self.exit_program)
        self.btnClose.clicked.connect(self.exit_program)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "All Files (*);;MP4 Files (*.mp4);;AVI Files (*.avi)", options=options)
        if fileName:
            self.pathVideoFile.setText(fileName)

    def showTime(self):
        self.DateTime = QDateTime.currentDateTime()
        self.lcdTime.display(self.DateTime.toString("hh:mm:ss"))

    def getCpuUsage(self, cpu):
        self.lblCPU.setText(str(cpu) + "%")
        if cpu > 15:
            self.lblCPU.setStyleSheet("color: rgb(23, 63, 95);")
        if cpu > 25:
            self.lblCPU.setStyleSheet("color: rgb(32, 99, 155);")
        if cpu > 45:
            self.lblCPU.setStyleSheet("color: rgb(60, 174, 163);")
        if cpu > 65:
            self.lblCPU.setStyleSheet("color: rgb(246, 213, 92);")
        if cpu > 85:
            self.lblCPU.setStyleSheet("color: rgb(237, 85, 59);")

    def getRamUsage(self, ram):
        self.lblRAM.setText(str(ram[2]) + "%")
        if ram[2] > 15:
            self.lblRAM.setStyleSheet("color: rgb(23, 63, 95);")
        if ram[2] > 25:
            self.lblRAM.setStyleSheet("color: rgb(32, 99, 155);")
        if ram[2] > 45:
            self.lblRAM.setStyleSheet("color: rgb(60, 174, 163);")
        if ram[2] > 65:
            self.lblRAM.setStyleSheet("color: rgb(246, 213, 92);")
        if ram[2] > 85:
            self.lblRAM.setStyleSheet("color: rgb(237, 85, 59);")

    def getFPS(self, fps):
        self.lblFPS.setText(str(fps))
        if fps > 5:
            self.lblFPS.setStyleSheet("color: rgb(237, 85, 59);")
        if fps > 15:
            self.lblFPS.setStyleSheet("color: rgb(60, 174, 155);")
        if fps > 25:
            self.lblFPS.setStyleSheet("color: rgb(85, 170, 255);")
        if fps > 35:
            self.lblFPS.setStyleSheet("color: rgb(23, 63, 95);")

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
        rgb_img = cv.cvtColor(src=Image, code=cv.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        cvt2QtFormat = QImage(rgb_img.data, w, h,
                              bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(cvt2QtFormat)
        return pixmap

    def StartWebCam(self):
        try:
            self.rdbVideoFile.setEnabled(False)
            self.listCam.setEnabled(False)
            self.txtLog.append(
                f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Start Webcam ({self.listCam.currentText()})")
            self.btnStop.setEnabled(True)
            self.btnStart.setEnabled(False)

            global sourceVideo
            sourceVideo = self.listCam.currentIndex()

            # Opencv QThread
            self.worker = ThreadClass()
            self.worker.imageUpdate.connect(self.opencv_emit)
            self.worker.fps.connect(self.getFPS)
            self.worker.start()

        except Exception as error:
            pass

    def StopWebCam(self):
        self.rdbVideoFile.setEnabled(True)
        self.listCam.setEnabled(True)
        self.txtLog.append(
            f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Stop Webcam ({self.listCam.currentText()})")
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        # Set Icon back to stop state
        self.worker.stop()

    def StartVideo(self):
        try:
            self.rdbCamera.setEnabled(False)
            self.btnSelect.setEnabled(False)
            self.txtLog.append(
                f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Start Video")
            self.btnStop.setEnabled(True)
            self.btnStart.setEnabled(False)

            global sourceVideo
            sourceVideo = self.pathVideoFile.text()

            # Opencv QThread
            self.worker = ThreadClass()
            self.worker.imageUpdate.connect(self.opencv_emit)
            self.worker.fps.connect(self.getFPS)
            self.worker.start()

        except Exception as error:
            pass

    def StopVideo(self):
        self.rdbCamera.setEnabled(True)
        self.btnSelect.setEnabled(True)
        self.txtLog.append(
            f"{self.DateTime.toString('hh:mm:ss dd/MM/yyyy')}: Stop Video")
        self.btnStart.setEnabled(True)
        self.btnStop.setEnabled(False)
        # Set Icon back to stop state
        self.worker.stop()

    def StartDetection(self):
        if self.rdbCamera.isChecked():
            self.StartWebCam()
        elif self.rdbVideoFile.isChecked():
            self.StartVideo()

    def StopDetection(self):
        if self.rdbCamera.isChecked():
            self.StopWebCam()
        elif self.rdbVideoFile.isChecked():
            self.StopVideo()

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
        global capture
        capture = cv.VideoCapture(sourceVideo)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)

        total_passed_vehicle = 0  # using it to count vehicles

        # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
        # What model to download.
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = os.path.join(os.path.dirname(
            __file__), MODEL_NAME, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        total_passed_vehicle = 0
        speed = 'waiting...'
        direction = 'waiting...'
        size = 'waiting...'
        color = 'waiting...'

        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name(
                    'detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')

                self.ThreadActive = True
                prev_frame_time = 0
                while self.ThreadActive:
                    (ret, frame_cap) = capture.read()
                    if not ret:
                        break
                    input_frame = frame_cap
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = \
                        sess.run([detection_boxes, detection_scores,
                                  detection_classes, num_detections],
                                 feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    (counter, csv_line) = \
                        vis_util.visualize_boxes_and_labels_on_image_array(
                        capture.get(1),
                        input_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                    )

                    total_passed_vehicle = total_passed_vehicle + counter

                    # insert information text to video frame
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(
                        input_frame,
                        'Detected Vehicles: ' + str(total_passed_vehicle),
                        (10, 35),
                        font,
                        0.8,
                        (0, 0xFF, 0xFF),
                        2,
                        cv.FONT_HERSHEY_SIMPLEX,
                    )

                    # when the vehicle passed over line and counted, make the color of ROI line green
                    if counter == 1:
                        cv.line(input_frame, (0, 200),
                                (640, 200), (0, 0xFF, 0), 5)
                    else:
                        cv.line(input_frame, (0, 200),
                                (640, 200), (0, 0, 0xFF), 5)

                    # insert information text to video frame
                    cv.rectangle(input_frame, (10, 275),
                                 (230, 337), (180, 132, 109), -1)
                    cv.putText(
                        input_frame,
                        'ROI Line',
                        (545, 190),
                        font,
                        0.6,
                        (0, 0, 0xFF),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        input_frame,
                        'LAST PASSED VEHICLE INFO',
                        (11, 290),
                        font,
                        0.5,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv.FONT_HERSHEY_SIMPLEX,
                    )
                    cv.putText(
                        input_frame,
                        '-Movement Direction: ' + direction,
                        (14, 302),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv.putText(
                        input_frame,
                        '-Speed(km/h): ' + str(speed).split(".")[0],
                        (14, 312),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv.putText(
                        input_frame,
                        '-Color: ' + color,
                        (14, 322),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv.putText(
                        input_frame,
                        '-Vehicle Size/Type: ' + size,
                        (14, 332),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                    if csv_line != 'not_available':
                        with open('traffic_measurement.csv', 'a') as f:
                            writer = csv.writer(f)
                            (size, color, direction, speed) = \
                                csv_line.split(',')
                            writer.writerows([csv_line.split(',')])

                    new_frame_time = time.time()
                    fpsValue = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    if ret:
                        self.imageUpdate.emit(input_frame)
                        self.fps.emit(int(fpsValue))

    def stop(self):
        self.ThreadActive = False
        self.fps.emit(0)
        self.quit()
