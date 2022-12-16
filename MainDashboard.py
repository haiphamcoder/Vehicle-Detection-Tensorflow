from PyQt5.QtWidgets import QMainWindow, QHeaderView, QFileDialog
from PyQt5.QtCore import QFile, QTimer, QDateTime, QThread, pyqtSignal, pyqtSlot, QAbstractTableModel
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtGui import QStandardItemModel, QImage, QPixmap, QStandardItem
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
from utils.visualization import Visualization

# Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
#  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings
NUM_PARALLEL_EXEC_UNITS = 4
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                                  allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)


def cvt_cv_qt(image):
    rgb_img = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    cvt2_qt_format = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(cvt2_qt_format)
    return pixmap


class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.date_time = None
        self.worker = None
        self.resource_usage = None
        self.lcd_timer = None
        self.load_ui()

    def load_ui(self):
        path = os.path.join(os.path.dirname(__file__), "MainUI.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()

        online_cam = QCameraInfo.availableCameras()
        self.listCam.addItems([c.description() for c in online_cam])
        self.btnStart.clicked.connect(self.StartDetection)
        self.btnStop.clicked.connect(self.StopDetection)
        self.btnStop.setEnabled(False)
        self.rdbModelMobilenetV1.setChecked(True)

        global table_model
        table_model = QStandardItemModel()
        header = ['Vehicle Type/Size', 'Vehicle Color', 'Vehicle Movement Direction', 'Vehicle Speed (km/h)']
        table_model.setHorizontalHeaderLabels(header)
        self.tblTraffic.setModel(table_model)
        self.tblTraffic.horizontalHeader().setStretchLastSection(True)
        self.tblTraffic.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.lcd_timer = QTimer()
        self.lcd_timer.timeout.connect(self.showTime)
        self.lcd_timer.start()

        self.resource_usage = SystemMonitor()
        self.resource_usage.start()
        self.resource_usage.cpu.connect(self.getCpuUsage)
        self.resource_usage.ram.connect(self.getRamUsage)

        self.rdbCamera.setChecked(True)
        self.btnSelect.setEnabled(False)
        self.pathVideoFile.setEnabled(False)
        self.rdbCamera.toggled.connect(lambda: self.buttonState(self.rdbCamera))
        self.rdbVideoFile.toggled.connect(lambda: self.buttonState(self.rdbVideoFile))
        self.btnSelect.clicked.connect(self.openFileNameDialog)

        global ckbDrawROI
        ckbDrawROI = self.ckbDrawROI
        ckbDrawROI.setChecked(True)
        global spinPosROI, spinX, spinY, spinW, spinH
        spinPosROI = self.spinPosROI
        spinPosROI.setValue(200)
        spinX = self.spinX
        spinX.setValue(0)
        spinY = self.spinY
        spinY.setValue(200)
        spinW = self.spinW
        spinW.setValue(640)
        spinH = self.spinH
        spinH.setValue(4)

        global ckbExportCSV
        ckbExportCSV = self.ckbExportCSV
        ckbExportCSV.setChecked(False)
        global ckbExportVideo
        ckbExportVideo = self.ckbExportVideo
        ckbExportVideo.setChecked(False)
        self.actionQuit.triggered.connect(self.exit_program)
        self.btnClose.clicked.connect(self.exit_program)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open", "",
                                                   "All Files (*);;MP4 Files (*.mp4);;AVI Files (*.avi)",
                                                   options=options)
        if file_name:
            self.pathVideoFile.setText(file_name)
            self.btnStart.setEnabled(True)

    def showTime(self):
        self.date_time = QDateTime.currentDateTime()
        self.lcdTime.display(self.date_time.toString("hh:mm:ss"))

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
            if button.isChecked():
                self.listCam.setEnabled(True)
                self.pathVideoFile.setEnabled(False)
                self.btnSelect.setEnabled(False)
                if self.listCam.count() == 0:
                    self.btnStart.setEnabled(False)
                else:
                    self.btnStart.setEnabled(True)
            else:
                self.listCam.setEnabled(False)
                self.pathVideoFile.setEnabled(True)
                self.btnSelect.setEnabled(True)
                if self.pathVideoFile.text() == "":
                    self.btnStart.setEnabled(False)
                else:
                    self.btnStart.setEnabled(True)

        if button.text() == "Video File":
            if button.isChecked():
                self.listCam.setEnabled(False)
                self.pathVideoFile.setEnabled(True)
                self.btnSelect.setEnabled(True)
                if self.pathVideoFile.text() == "":
                    self.btnStart.setEnabled(False)
                else:
                    self.btnStart.setEnabled(True)
            else:
                self.listCam.setEnabled(True)
                self.pathVideoFile.setEnabled(False)
                self.btnSelect.setEnabled(False)
                if self.listCam.count() == 0:
                    self.btnStart.setEnabled(False)
                else:
                    self.btnStart.setEnabled(True)

    @pyqtSlot(np.ndarray)
    def opencv_emit(self, image):
        original = cvt_cv_qt(image)
        self.mainFrame.setPixmap(original)
        self.mainFrame.setScaledContents(True)

    def StartDetection(self):
        global widthVid, heightVid, fpsVid, source_video, output_movie, capture
        self.groupSource.setEnabled(False)
        self.groupModels.setEnabled(False)
        self.groupOption.setEnabled(False)
        if self.rdbCamera.isChecked():
            self.rdbVideoFile.setEnabled(False)
            self.listCam.setEnabled(False)
            self.txtLog.append(
                f"{self.date_time.toString('hh:mm:ss dd/MM/yyyy')}: Start Webcam ({self.listCam.currentText()})")
            self.btnStop.setEnabled(True)
            self.btnStart.setEnabled(False)
            source_video = self.listCam.currentIndex()

        elif self.rdbVideoFile.isChecked():
            self.rdbCamera.setEnabled(False)
            self.btnSelect.setEnabled(False)
            self.txtLog.append(f"{self.date_time.toString('hh:mm:ss dd/MM/yyyy')}: Start Video")
            self.btnStop.setEnabled(True)
            self.btnStart.setEnabled(False)
            source_video = self.pathVideoFile.text()

        capture = cv.VideoCapture(source_video)
        widthVid = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        heightVid = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        fpsVid = int(capture.get(cv.CAP_PROP_FPS))

        if self.ckbExportCSV.isChecked():
            with open('traffic_measurement.csv', 'w') as f:
                writer = csv.writer(f)
                csv_line = 'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
                writer.writerows([csv_line.split(',')])
        if self.ckbExportVideo.isChecked():
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            output_movie = cv.VideoWriter('output.avi', fourcc, fpsVid, (widthVid, heightVid))

        # Opencv QThread
        self.worker = ThreadClass()
        self.worker.imageUpdate.connect(self.opencv_emit)
        self.worker.fps.connect(self.getFPS)
        self.worker.start()

    def StopDetection(self):
        self.groupSource.setEnabled(True)
        self.groupModels.setEnabled(True)
        self.groupOption.setEnabled(True)
        if self.rdbCamera.isChecked():
            self.rdbVideoFile.setEnabled(True)
            self.listCam.setEnabled(True)
            self.txtLog.append(
                f"{self.date_time.toString('hh:mm:ss dd/MM/yyyy')}: Stop Webcam ({self.listCam.currentText()})")
            self.btnStart.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.worker.stop()
        elif self.rdbVideoFile.isChecked():
            self.rdbCamera.setEnabled(True)
            self.btnSelect.setEnabled(True)
            self.txtLog.append(f"{self.date_time.toString('hh:mm:ss dd/MM/yyyy')}: Stop Video")
            self.btnStart.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.worker.stop()

    def exit_program(self):
        self.ThreadActive = False
        sys.exit()


class SystemMonitor(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.ThreadActive = None

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
    ThreadActive = False

    def run(self):
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)

        # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
        # for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
        # What model to download.
        model_name = 'ssd_mobilenet_v1_coco_2018_01_28'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        path_to_ckpt = os.path.join(os.path.dirname(__file__), model_name, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')

        num_classes = 90

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts 5, we know that
        # this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary
        # mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        total_passed_vehicle = 0
        speed = 'waiting...'
        direction = 'waiting...'
        size = 'waiting...'
        color = 'waiting...'

        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                self.ThreadActive = True
                prev_frame_time = 0
                while self.ThreadActive:
                    (ret, frame_cap) = capture.read()
                    if not ret:
                        capture.release()
                        break
                    input_frame = frame_cap
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    vis = Visualization()
                    vis.ROI_POSITION = spinPosROI.value()
                    (counter, csv_line) = vis.visualize_boxes_and_labels_on_image_array(capture.get(1), input_frame,
                                                                                        np.squeeze(boxes),
                                                                                        np.squeeze(classes).astype(
                                                                                            np.int32),
                                                                                        np.squeeze(scores),
                                                                                        category_index,
                                                                                        use_normalized_coordinates=True,
                                                                                        line_thickness=4, )

                    total_passed_vehicle = total_passed_vehicle + counter

                    # insert information text to video frame
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(input_frame, 'Detected Vehicles: ' + str(total_passed_vehicle), (10, 35), font, 0.8,
                               (0, 0xFF, 0xFF), 2, cv.FONT_HERSHEY_SIMPLEX, )

                    # when the vehicle passed over line and counted, make the color of ROI line green
                    if ckbDrawROI.isChecked():
                        if counter == 1:
                            cv.line(input_frame, (spinX.value(), spinY.value()), (spinW.value(), spinY.value()),
                                    (0, 0xFF, 0), 3)
                        else:
                            cv.line(input_frame, (spinX.value(), spinY.value()), (spinW.value(), spinY.value()),
                                    (0, 0, 0xFF), 3)

                        cv.putText(input_frame, 'ROI Line', (spinW.value() - 100, spinY.value() - 10), font, 0.6,
                                   (0, 0, 0xFF), 2, cv.LINE_AA, )

                    # insert information text to video frame
                    cv.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)

                    cv.putText(input_frame, 'LAST PASSED VEHICLE INFO', (11, 290), font, 0.5, (0xFF, 0xFF, 0xFF), 1,
                               cv.FONT_HERSHEY_SIMPLEX, )
                    cv.putText(input_frame, '-Movement Direction: ' + direction, (14, 302), font, 0.4,
                               (0xFF, 0xFF, 0xFF), 1, cv.FONT_HERSHEY_COMPLEX_SMALL, )
                    cv.putText(input_frame, '-Speed(km/h): ' + str(speed).split(".")[0], (14, 312), font, 0.4,
                               (0xFF, 0xFF, 0xFF), 1, cv.FONT_HERSHEY_COMPLEX_SMALL, )
                    cv.putText(input_frame, '-Color: ' + color, (14, 322), font, 0.4, (0xFF, 0xFF, 0xFF), 1,
                               cv.FONT_HERSHEY_COMPLEX_SMALL, )
                    cv.putText(input_frame, '-Vehicle Size/Type: ' + size, (14, 332), font, 0.4, (0xFF, 0xFF, 0xFF), 1,
                               cv.FONT_HERSHEY_COMPLEX_SMALL, )

                    if csv_line != 'not_available':
                        (size, color, direction, speed) = csv_line.split(',')
                        infor_vehicle = [QStandardItem(size), QStandardItem(color), QStandardItem(direction),
                                         QStandardItem(speed)]
                        table_model.insertRow(table_model.rowCount(), infor_vehicle)
                        if ckbExportCSV.isChecked():
                            with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)
                                writer.writerows([csv_line.split(',')])

                    if ckbExportVideo.isChecked():
                        output_movie.write(input_frame)
                    new_frame_time = time.time()
                    fpsValue = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    if ret:
                        self.imageUpdate.emit(input_frame)
                        self.fps.emit(int(fpsValue))

                capture.release()
                self.stop()

    def stop(self):
        self.ThreadActive = False
        self.fps.emit(0)
        self.quit()
