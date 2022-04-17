import cv2
import mainwindow
from PyQt5 import QtWidgets
from PyQt5.Qt import QTime, QTimer
from PyQt5.QtGui import QImage, QPixmap, qRgb
from PyQt5.QtWidgets import QFileDialog
from utils_photo_feature import *

sys.path.append("../photo_service_s3fd")
import net_s3fd

gray_color_table = [qRgb(i, i, i) for i in range(256)]
NUM_CLASSES = 1
GENDER_CLASSES = ["Male", "Female"]


def convert_ndarray_to_qimg(arr):
    if arr is None:
        return QImage()
    qim = None
    if arr.dtype is not np.uint8:
        arr = arr.astype(np.uint8)
    if arr.dtype == np.uint8:
        if len(arr.shape) == 2:
            qim = QImage(
                arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_Indexed8
            )
            qim.setColorTable(gray_color_table)
        elif len(arr.shape) == 3:
            if arr.shape[2] == 3:
                qim = QImage(
                    arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888
                )
    return qim.copy()


class quiApp(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.video_path = "Silicon.Valley.S05E07.WEB-DL.1080p.Rus.Eng.Subs.mkv"
        self.video_capturer = cv2.VideoCapture(self.video_path)
        self.video_status = "Video is not playing"
        self.width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.video_path[:-4] + "_label.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            25.0,
            (self.width, self.height),
        )
        self.video_frame = None
        self.video_src = "camera"
        self.proc_image_width = 960
        self.proc_image_height = 540
        self.proc_qimage = QImage()
        self.proc_qpixmap = QPixmap()
        self.tmr = QTimer(self)
        self.tmr.setInterval(40)
        self.tmr.timeout.connect(self.timeout_slot)
        self.tmr.start()
        self.time = QTime()
        self.time.start()
        self.fps_period = None
        self.last_timestamp = self.time.elapsed()
        self.enable_detector = False
        self.enable_detector_s3fd = False
        self.enable_other_features = False
        self.enable_playing = False
        self.enable_record = False

        self.model_face_detector_path = "graph_mobilenet_v2_all_180627.pb"
        self.model_face_detector_s3fd = "s3fd_convert.pth"
        self.gender_model_path = "model_25_epoch_bi.pt"
        self.age_model_path = "model_37_epoch.pt"

        self.ssd = None
        self.s3fd = None
        self.model_gender = None
        self.model_age = None

        self.camera_id = 0
        self.camera_capturer = cv2.VideoCapture(self.camera_id)
        self.camera_status = "Stream is not active"
        self.camera_frame = None

        self.amount_of_frames = None

        self.setWindowTitle("Face detector with other fetures")

        self.checkBox_enable_detection.toggled.connect(self.checkBox_enable_detection_toggled)
        self.checkBox_enable_detection_s3fd.toggled.connect(
            self.checkBox_enable_detection_s3fd_toggled
        )
        self.checkBox_enable_playing.toggled.connect(self.checkBox_enable_playing_toggled)
        self.checkBox_enable_camera.toggled.connect(self.checkBox_enable_camera_toggled)
        self.checkBox_enable_other_features.toggled.connect(
            self.checkBox_enable_other_features_toggled
        )
        self.checkBox_enable_record.toggled.connect(self.checkBox_enable_record_toggled)

        self.lineEdit_path_video.editingFinished.connect(self.lineEdit_path_video_editing_finished)
        self.lineEdit_path_graph.editingFinished.connect(self.lineEdit_path_graph_editing_finished)
        self.pushButton_load_video.clicked.connect(self.pushButton_load_video_clicked)
        self.pushButton_load_graph.clicked.connect(self.pushButton_load_graph_clicked)
        self.lineEdit_path_graph.setText(self.model_face_detector_path)
        self.lineEdit_path_video.setText(self.video_path)
        self.lineEdit_path_video_editing_finished()
        self.lineEdit_path_gender_model.setText(self.gender_model_path)
        self.lineEdit_path_age_model.setText(self.age_model_path)
        self.checkBox_enable_camera.setChecked(True)

        self.horizontalSlider_video.sliderMoved.connect(self.horizontalSlider_video_moved)
        self.label_cur_frame.setText("00:00")
        self.cur_frame_id = 0

        self.output = None

        pass

    def timeout_slot(self):
        if self.video_src == "camera":
            self.refresh_camera_frame()
            self.proc_frame = self.camera_frame
        if self.video_src == "video":
            self.cur_frame_id += 1
            self.refresh_video_frame()
            self.proc_frame = self.video_frame
        if self.proc_frame is not None:
            if self.enable_detector:
                self.output = self.ssd.run_inference_for_single_image(self.proc_frame)
            elif self.enable_detector_s3fd:
                self.output = run_inference_s3fd(self.proc_frame, self.s3fd)
            if self.output is not None:
                bboxes = get_bboxes_face_detector(self.output)
                if len(bboxes) != 0:
                    for bbox in bboxes:
                        if self.enable_detector:
                            ymin, xmin, ymax, xmax = bbox
                        elif self.enable_detector_s3fd:
                            xmin, ymin, xmax, ymax = bbox
                        else:
                            xmin = None
                        if xmin is not None:
                            if self.proc_frame.shape[1] > 640:
                                thickness = 5
                            else:
                                thickness = 2
                            cv2.rectangle(
                                self.proc_frame,
                                (
                                    int(xmin * self.proc_frame.shape[1]),
                                    int(ymin * self.proc_frame.shape[0]),
                                ),
                                (
                                    int(xmax * self.proc_frame.shape[1]),
                                    int(ymax * self.proc_frame.shape[0]),
                                ),
                                (255, 0, 0),
                                thickness,
                            )
                            if self.enable_other_features:
                                face_crop = self.proc_frame[
                                    int(ymin * self.proc_frame.shape[0]) : int(
                                        ymax * self.proc_frame.shape[0]
                                    ),
                                    int(xmin * self.proc_frame.shape[1]) : int(
                                        xmax * self.proc_frame.shape[1]
                                    ),
                                ]
                                predict_gender = model_forward_gender(face_crop, self.model_gender)
                                cv2.putText(
                                    self.proc_frame,
                                    GENDER_CLASSES[predict_gender],
                                    (
                                        int(xmin * self.proc_frame.shape[1]),
                                        int(ymin * self.proc_frame.shape[0]) - 5,
                                    ),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    thickness // 2,
                                    (0, 230, 0),
                                    thickness // 2,
                                    cv2.LINE_AA,
                                )
                            else:
                                cv2.putText(
                                    self.proc_frame,
                                    "face",
                                    (
                                        int(xmin * self.proc_frame.shape[1]),
                                        int(ymin * self.proc_frame.shape[0]) - 5,
                                    ),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    thickness // 2,
                                    (0, 230, 0),
                                    thickness // 2,
                                    cv2.LINE_AA,
                                )

            if self.enable_record:
                self.out.write(cv2.cvtColor(self.proc_frame, cv2.COLOR_RGB2BGR))
            self.proc_frame = cv2.resize(
                self.proc_frame,
                (self.proc_image_width, self.proc_image_height),
                interpolation=cv2.INTER_CUBIC,
            )
            self.proc_qimage = convert_ndarray_to_qimg(self.proc_frame)
            self.proc_qpixmap = QPixmap.fromImage(self.proc_qimage)
            if self.proc_qpixmap is not None:
                self.label_image.setPixmap(self.proc_qpixmap)
            if self.video_src == "video":
                self.horizontalSlider_video.setSliderPosition(
                    self.cur_frame_id * 100 // self.amount_of_frames
                )
                self.horizontalSlider_time(self.cur_frame_id)

        cur_time = self.time.elapsed()
        self.fps_period = cur_time - self.last_timestamp
        self.last_timestamp = cur_time
        self.label_processing_time.setText(str(self.fps_period))
        self.label_fps.setText(str(int(1000.0 / self.fps_period)))
        pass

    def refresh_camera_frame(self):
        ret, self.camera_frame = self.camera_capturer.read()
        if self.camera_frame is not None:
            self.camera_frame = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
            self.camera_status = "Capturing in progress"
        else:
            self.camera_status = "Error"
        pass

    def refresh_camera_stream(self):
        self.camera_capturer.release()
        self.camera_capturer = cv2.VideoCapture(self.camera_id)
        pass

    def refresh_video_frame(self):
        ret, self.video_frame = self.video_capturer.read()
        if self.video_frame is not None:
            self.video_frame = cv2.cvtColor(self.video_frame, cv2.COLOR_BGR2RGB)
            self.video_status = "Playing in progress"
        else:
            self.video_status = "Error"
        self.label_video_status.setText(self.video_status)
        pass

    def refresh_video_stream(self):
        self.video_capturer.release()
        self.video_capturer = cv2.VideoCapture(self.video_path)
        pass

    def lineEdit_path_video_editing_finished(self):
        self.video_path = self.lineEdit_path_video.text()
        self.width = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.video_path[:-4] + "_label.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            25.0,
            self.refresh_video_stream(),
        )
        self.video_capturer = cv2.VideoCapture(self.video_path)
        self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.amount_of_frames = self.video_capturer.get(cv2.CAP_PROP_FRAME_COUNT)
        self.refresh_video_stream()
        pass

    def lineEdit_path_graph_editing_finished(self):
        self.model_face_detector_path = self.lineEdit_path_graph.text()
        pass

    def checkBox_enable_other_features_toggled(self, checked):
        if self.model_gender is None:
            self.model_gender = init_gender_model("model_25_epoch_bi.pt", "cuda:0")
        if self.model_age is None:
            self.model_age = init_age_model("model_37_epoch.pt", "cuda:0")
        self.enable_other_features = checked
        pass

    def checkBox_enable_detection_toggled(self, checked):
        # if checked:
        if self.ssd is not tf_classifier:
            self.ssd = tf_classifier()
            self.ssd.select_path_to_model(self.model_face_detector_path)
        # else:
        #     self.ssd.close_sess()
        self.enable_detector = checked
        pass

    def checkBox_enable_detection_s3fd_toggled(self, checked):
        if self.s3fd is None:
            self.s3fd = net_s3fd.s3fd()
            self.s3fd.load_state_dict(torch.load(self.model_face_detector_s3fd))
            self.s3fd.cuda()
            self.s3fd.eval()
        self.enable_detector_s3fd = checked
        pass

    def checkBox_enable_camera_toggled(self, checked):
        if checked:
            self.video_src = "camera"
        else:
            self.video_src = "none"
        self.switch_video_src(self.video_src)
        pass

    def checkBox_enable_playing_toggled(self, checked):
        if checked:
            self.video_src = "video"
        else:
            self.video_src = "none"
        self.switch_video_src(self.video_src)
        pass

    def checkBox_enable_record_toggled(self, checked):
        self.enable_record = checked
        pass

    def switch_video_src(self, new_src):
        if new_src == "camera":
            self.checkBox_enable_playing.setChecked(False)
            self.checkBox_enable_camera.setChecked(True)
        elif new_src == "video":
            self.checkBox_enable_playing.setChecked(True)
            self.checkBox_enable_camera.setChecked(False)
        else:
            self.checkBox_enable_playing.setChecked(False)
            self.checkBox_enable_camera.setChecked(False)
        pass

    def pushButton_load_video_clicked(self):
        fname = QFileDialog.getOpenFileName(
            self, "Select video", "", "", options=QFileDialog.DontUseNativeDialog
        )[0]
        self.lineEdit_path_video.setText(fname)
        self.video_path = self.lineEdit_path_video.text()
        self.refresh_video_stream()
        pass

    def pushButton_load_graph_clicked(self):
        fname = QFileDialog.getOpenFileName(
            self, "Select graph SSD", "", "", options=QFileDialog.DontUseNativeDialog
        )[0]
        self.lineEdit_path_graph.setText(fname)
        pass

    def horizontalSlider_video_moved(self):
        self.cur_frame_id = int(
            self.amount_of_frames * self.horizontalSlider_video.sliderPosition() / 100
        )
        self.video_capturer.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_id)
        self.refresh_video_frame()
        self.horizontalSlider_time(self.cur_frame_id)
        pass

    def horizontalSlider_time(self, cur_frame_id):
        if cur_frame_id > 1500:
            minutes = int(cur_frame_id // 1500)
            seconds = int((cur_frame_id % 1500) // 25)
        else:
            minutes = 0
            seconds = int(cur_frame_id // 25)
        str_seconds = "{0:02}".format(seconds)
        str_minutes = "{0:02}".format(minutes)
        self.label_cur_frame.setText(str_minutes + ":" + str_seconds)
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = quiApp()
    window.show()
    sys.exit(app.exec_())
    pass


if __name__ == "__main__":
    main()
