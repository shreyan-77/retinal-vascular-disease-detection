import cv2
import numpy as np

def is_retinal_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    h, w = img.shape[:2]

    # -------- 1️⃣ Circle Detection --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=50,
        param2=30,
        minRadius=int(min(h,w)*0.25),
        maxRadius=int(min(h,w)*0.6)
    )

    circle_detected = circles is not None


    # -------- 2️⃣ Dark Border Check --------
    top = gray[:20,:]
    bottom = gray[-20:,:]
    left = gray[:,:20]
    right = gray[:,-20:]

    border_mean = (
        np.mean(top) +
        np.mean(bottom) +
        np.mean(left) +
        np.mean(right)
    ) / 4

    dark_border = border_mean < 70


    # -------- 3️⃣ Red Dominance Check --------
    b,g,r = cv2.split(img)

    red_dominance = np.mean(r) > np.mean(b)


    # -------- Final Decision --------
    if circle_detected and dark_border and red_dominance:
        return True

    return False