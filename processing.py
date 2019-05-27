import numpy as np
import cv2 as cv
import imutils
import math
from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema


def midpoint(pta, ptb):
    return (pta[0] + ptb[0]) * 0.5, (pta[1] + ptb[1]) * 0.5


def get_nice_limits(img):
    hsv1 = img[int(img.shape[0] / 2)][0]
    hsv2 = img[int(img.shape[0] / 2)][img.shape[1] - 1]
    hsv3 = img[0][int(img.shape[1] / 2)]
    hsv4 = img[0][int(img.shape[1] / 2)]
    margin = 70
    bottom = (0,
              np.int(max(min(hsv1[1], hsv2[1], hsv3[1], hsv4[1]) - margin, 0)),
              np.int(max(min(hsv1[2], hsv2[2], hsv3[2], hsv4[2]) - margin, 0)))
    top = (255,
           255,
           np.int(min(max(hsv1[2], hsv2[2], hsv3[2], hsv4[2]) + margin, 255)))

    return bottom, top


def find_intersection(line1, line2):

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    if math.fabs(theta1 - theta2) < 0.1:
        return [0, 0]
    else:
        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]


image_show_ratio = 3
line_thickness = 2


class Card:
    def __init__(self, image):
        self.img = image
        self.bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.contours = []
        self.external_contour = []
        self.tilt = 0
        self.is_tilted = False
        self.rank = 0
        self.area = 0
        self.middle = [0, 0]
        self.middles_of_marks = []
        self.corners = []

    def find_corners(self):
        orig = self.img.copy()
        grey = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(grey, 50, 150, apertureSize=3)
        lines = cv.HoughLines(canny, 1, np.pi / 360, 100)
        intersections = []
        for i in lines:
            for j in lines:
                x_found, y_found = find_intersection(i, j)
                is_unique = True
                for [x_int, y_int] in intersections:
                    if math.fabs(x_found - x_int) < 50 and math.fabs(y_found - y_int) < 50:
                        is_unique = False
                if is_unique:
                    if [x_found, y_found] != [0, 0]:
                        intersections.append([x_found, y_found])
                        cv.circle(orig, (x_found, y_found), 5, (0, 0, 255), -1)
        self.corners = intersections

        # imS = cv.resize(orig, (np.int(1936 / image_show_ratio), np.int(1216 / image_show_ratio)))
        # cv.imshow("rysuneczekk", imS)

    def edges_middle_area_tilt(self):

        for c in self.contours:
            if cv.contourArea(c) > 10000:
                # orig = self.img.copy()
                self.area = cv.contourArea(c)
                self.external_contour = c

                box = cv.minAreaRect(c)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                # cv.drawContours(orig, c, -1, (0, 255, 0), 5)

                (tl, tr, br, bl) = box
                self.tilt = math.fabs((tr[0] - tl[0]) / (tr[1] - tl[1]))
                if 5 > self.tilt > 0.5:
                    self.is_tilted = True
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                mX, mY = midpoint((tltrX, tltrY), (blbrX, blbrY))
                self.middle = (np.int(mX), np.int(mY))
                # cv.circle(orig, self.middle, 5, (0, 0, 255), -1)

                # imS = cv.resize(orig, (np.int(1936/image_show_ratio), np.int(1216/image_show_ratio)))
                # cv.imshow("rysuneczek", imS)

    def get_rank(self):
        # orig = self.img.copy()
        for c in self.contours:
            if 450 < cv.contourArea(c) < 10000:
                self.rank += 1
                box = cv.minAreaRect(c)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)

                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                mX, mY = midpoint((tltrX, tltrY), (blbrX, blbrY))
                center_point = (np.int(mX), np.int(mY))
                #cv.circle(orig, center_point, 5, (255, 0, 0), -1)
                self.middles_of_marks.append(center_point)

        # imS = cv.resize(orig, (np.int(1936/image_show_ratio), np.int(1216/image_show_ratio)))
        # cv.imshow("rysuneczek", imS)

    def processing(self):
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        hsv_v = hsv[:, :, 2]

        # plt.hist(hsv_v.ravel(), 255, [1, 256])
        # plt.show()

        h = np.histogram(hsv_v.ravel(), bins=range(0, 256))
        hdata = h[0][1:]
        peaks_all = argrelextrema(hdata, np.greater, order=10)
        peaks = []
        for i in peaks_all[0]:
            if hdata[i] > 2000:
                peaks.append(i)

        self.bw = cv.inRange(self.bw, np.median(peaks)-35, 255) #150, 255

        # imS = cv.resize(self.bw, (np.int(1936 / image_show_ratio), np.int(1216 / image_show_ratio)))
        # cv.imshow("card bw", imS)

        self.contours = cv.findContours(self.bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        self.contours = imutils.grab_contours(self.contours)
        (self.contours, _) = contours.sort_contours(self.contours)

        self.find_corners()
        self.edges_middle_area_tilt()
        self.get_rank()

    def task_tilt(self, image):
        if self.is_tilted:
            cv.drawContours(image, self.external_contour, -1, (0, 0, 150), line_thickness)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(image, str(self.area), self.middle, font, 1.2, (0, 0, 150), line_thickness, cv.LINE_AA)

    def find_colors_in_2_or_7(self, image, middle):
        if self.rank == 2 or self.rank == 7:
            for i in self.middles_of_marks:
                cv.line(image, middle, i, (255, 100, 100), line_thickness)

    def draw_rectangle(self, image):
        if self.rank == 9 or self.rank == 8:
            self.corners = np.array(self.corners, dtype=int)
            self.corners = perspective.order_points(self.corners)
            cv.drawContours(image, [self.corners.astype("int")], -1, (0, 255, 0), line_thickness)


class Picture:
    def __init__(self, image, name):
        self.img = image
        self.name = name
        self.marked_image = self.img.copy()
        self.cards = self.img.copy()
        self.segmented = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.markers = 0
        self.list_of_cards = []
        self.middle_of_picture = (int(self.img.shape[1] / 2), int(self.img.shape[0] / 2))

    def segmentation(self):
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        bottom, top = get_nice_limits(hsv)
        background_threshold = cv.inRange(hsv, bottom, top)
        self.cards = cv.bitwise_not(background_threshold)
        self.cards = cv.medianBlur(self.cards, 13)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
        self.cards = cv.morphologyEx(self.cards, cv.MORPH_CLOSE, kernel)

        ret, self.markers = cv.connectedComponents(self.cards)
        for i in range(5):
            val = 20*i
            self.segmented[self.markers == i] = val

        # imS = cv.resize(self.cards, (np.int(1936/image_show_ratio), np.int(1216/image_show_ratio)))
        # cv.imshow("cards", imS)

    def extraction(self):
        for i in range(1, 5):
            card_mask = self.markers.copy()
            card_mask[self.markers != i] = 0
            card_pic = self.img.copy()
            card_pic[card_mask == 0] = [0, 0, 0]
            card = Card(card_pic)
            self.list_of_cards.append(card)

    def run_processing(self):
        self.segmentation()
        self.extraction()
        for i in self.list_of_cards:
            i.processing()

    def run_tasks(self):
        for i in self.list_of_cards:
            i.task_tilt(self.marked_image)
            i.find_colors_in_2_or_7(self.marked_image, self.middle_of_picture)
            i.draw_rectangle(self.marked_image)
            print(i.rank)

        imS = cv.resize(self.marked_image, (np.int(1936/image_show_ratio), np.int(1216/image_show_ratio)))
        cv.imshow(self.name, imS)
