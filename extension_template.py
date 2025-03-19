from krita import *
from PyQt5.QtWidgets import (QAction, QMessageBox,QWidget,QMessageBox,QDialog,QPushButton,QHBoxLayout,QLabel
                             ,QFrame,QVBoxLayout)
from PyQt5.QtCore import Qt,QTimer,qDebug
from PyQt5.QtGui import QCursor
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os

point_selected = False
point = None
camera = cv2.VideoCapture(1)

def log(message):
    layoutForButtons = QHBoxLayout()
    label = QLabel(str(message))

    label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
    label.setAlignment(Qt.AlignBottom | Qt.AlignRight)
    layoutForButtons.addWidget(label)

    # create dialog  and show it
    newDialog = QDialog()
    newDialog.setLayout(layoutForButtons)
    newDialog.setWindowTitle("New Dialog Title!")
    newDialog.exec_()  # show the dialog

def trackerCSRT():
    # Create MIL tracker
    tracker = cv2.TrackerMIL.create()

    # Read the first frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        camera.release()
        exit()

    # Select the ROI (Region of Interest) manually
    bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)

    # Initialize the tracker with the selected region
    tracker.init(frame, bbox)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Track the center point
            QCursor.setPos(center_x, center_y)
            print(f"Tracked Point: X={center_x}, Y={center_y}")  # Output coordinates

        # Display the frame
        cv2.imshow("Tracking", frame)

        # Exit when 'Esc' key is pressed
        if cv2.waitKey(20) & 0xFF == 27:
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()


def openCVTest(self):
    global point, point_selected

    # Mouse callback function to select a point
    def select_point(event, x, y, flags, param):
        global point, point_selected
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            point = np.array([[x, y]], dtype=np.float32)  # Store the clicked point
            point_selected = True  # Mark that a point has been selected
            qDebug(f"Point Selected: {x}, {y}")  # Debugging to confirm click


    # Initialize video capture
    cap = cv2.VideoCapture(1)

    # Create a window and set the mouse callback function
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", select_point)

    # Read the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If a point has been selected, track it
        if point_selected and point is not None:

            # Compute optical flow to track the point
            new_point, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, point.reshape(-1, 1, 2), None, **lk_params)

            # If tracking is successful, update the point position
            if new_point is not None and st[0][0] == 1:
                point = new_point
                a, b = point.ravel()
                cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)  # Draw tracked point
                qDebug(f"Tracked Point: X={int(a)}, Y={int(b)}")  # Print coordinates

                QCursor.setPos(int(a), int(b))

            # Update previous frame for next iteration
            old_gray = gray.copy()

        # Show the video with tracking
        cv2.imshow("Tracking", frame)

        # Exit if the user presses 'Esc'
        if cv2.waitKey(30) & 0xFF == 27:
            point = None
            point_selected = False
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


class ExtensionTemplate(Extension):

    def __init__(self, parent):
        super().__init__(parent)
        self.target_x = 300  # Target X coordinate
        self.target_y = 300  # Target Y coordinate
        self.step_size = 5  # Pixels per step
        self.timer = QTimer()
        self.cursor = QCursor()
        self.timer.timeout.connect(self.cursorMove)

    # Krita.instance() exists, so do any setup work
    def setup(self):
        pass

    def startcursorMove(self):
        self.current_x = self.cursor.pos().x()
        self.current_y = self.cursor.pos().y()
        self.timer.start(10)

    def cursorMove(self):
        if abs(self.current_x - self.target_x) < self.step_size and abs(self.current_y - self.target_y) < self.step_size:

            self.timer.stop()
            return

        # Redo the math so it go in diagonal
        if self.current_x < self.target_x:
            self.current_x += self.step_size
        elif self.current_x > self.target_x:
            self.current_x -= self.step_size

        if self.current_y < self.target_y:
            self.current_y += self.step_size
        elif self.current_y > self.target_y:
            self.current_y -= self.step_size

        QCursor.setPos(self.current_x, self.current_y)

    def main(self):
        # QMessageBox creates quick popup with information
        '''
        messageBox = QMessageBox()
        messageBox.setInformativeText("Open cv project")
        messageBox.setWindowTitle('Opencv')
        messageBox.setText("Hello! Here is the version of Krita you are using.");
        messageBox.setStandardButtons(QMessageBox.Close)
        messageBox.setIcon(QMessageBox.Information)
        messageBox.exec()
        '''

        layoutForButtons = QHBoxLayout()
        newButton = QPushButton("Press me")
        label = QLabel("Hello World!")

        label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        label.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        newButton.clicked.connect(trackerCSRT)
        layoutForButtons.addWidget(newButton)
        layoutForButtons.addWidget(label)

        # create dialog  and show it
        newDialog = QDialog()
        newDialog.setLayout(layoutForButtons)
        newDialog.setWindowTitle("New Dialog Title!")
        newDialog.exec_()  # show the dialog


    def createActions(self, window):
        action = window.createAction("", "OpencvTest")
        action.triggered.connect(self.main)