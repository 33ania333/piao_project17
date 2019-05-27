import processing
import cv2 as cv

img = cv.imread("szum.jpg", 3)
obraz = processing.Picture(img, "szum")
obraz.run_processing()
obraz.run_tasks()

img = cv.imread("ideal.jpg", 3)
obraz = processing.Picture(img, "ideal")
obraz.run_processing()
obraz.run_tasks()

img = cv.imread("blur.jpg", 3)
obraz = processing.Picture(img, "blur")
obraz.run_processing()
obraz.run_tasks()

img = cv.imread("gradient.jpg", 3)
obraz = processing.Picture(img, "gradient")
obraz.run_processing()
obraz.run_tasks()

cv.waitKey(0)
