import os
import numpy as np
import scipy.signal
import math as m
from PIL import Image
import matplotlib.pyplot as plt

def convert_to_grayscale(image):
    return np.array(image.convert("L"))



def detect_cm_marks(gray_array):
    def find(line):
        marks, prev = [], False
        for i, val in enumerate(line):
            curr = val > 10
            dist = True
            if marks: dist = i - marks[-1] > 10
            if curr and not prev and dist:
                marks.append(i) 
            prev = curr
        print(marks)
        return marks
    
    rows = find(gray_array[:,2])
    cols = find(gray_array[2])
    d = ((rows[-1] - rows[1]) + (cols[-1] - cols[1])) / (len(rows) + len(cols) - 4)
    print(d)

    return d, rows, cols



def clean_image(gray_array):
    idx = np.where(np.sum(gray_array, axis=1) < 500)[0][0]
    img = gray_array[idx:, 6:].astype(float)
    img[(img > 200) | (img < 5)] = 0
    img[:100,:100], img[:100,-100:] = 0, 0

    return img, idx



def calculate_geometry(temp, d, alpha):
    row = np.where(np.sum(temp, axis = 1) > 200)[0][0]
    peaks = scipy.signal.find_peaks(temp[row], distance=50)[0]
    cln = peaks[abs(peaks) <= len(temp[0]) * 4 / 5]
    d_cm = abs(cln[0] - cln[1]) / (2 * d)
    offset_cm = d_cm / m.sin(alpha)
    r = np.where(temp[:, int(cln[0] + abs(cln[0] - cln[1]) / 2)] > 0)[0]

    return offset_cm * d, r[-1] - r[0], offset_cm, (r[-1] - r[0]) / d, r[0]


def compute_coordinate_grid(offset_cm, r_cm, d, temp, offset_px, first, alpha_deg, res):
    rows = np.arange(int(offset_cm*res), int(offset_cm*res + r_cm*res))[:, None]
    cols = np.linspace(-alpha_deg, alpha_deg, len(rows))
    Th = cols * m.pi / 180
    X, Y = np.sin(Th) * rows, np.cos(Th) * rows

    return X / res * d + temp.shape[1]/2, Y / res * d - (offset_px - first)



def trilinear_interpolation(X, Y, temp):
    X = np.clip(X, 0, temp.shape[0]-2)
    Y = np.clip(Y, 0, temp.shape[1]-2)

    Xl = np.floor(X).astype(int)
    Xr = Xl + 1
    Yt = np.floor(Y).astype(int)
    Yb = Yt + 1

    dx = X - Xl
    dy = Y - Yt

    intensity = (
        (1 - dx) * (1 - dy) * temp[Yt, Xl] +
        (dx * (1 - dy)) * temp[Yt, Xr] +
        (1 - dx) * dy * temp[Yb, Xl] +
        (dx * dy) * temp[Yb, Xr]
    )

    return np.clip(intensity, 0, 255)


def transform(path, alpha_deg, res):
    gray = convert_to_grayscale(Image.open(path))
    d, rows, cols = detect_cm_marks(gray)
    #alpha_deg = 31.5
    alpha = alpha_deg/180*m.pi
    temp, gray_idx = clean_image(gray)
    offset_px, r_px, offset_cm, r_cm, first = calculate_geometry(temp, d, alpha)

    X, Y = compute_coordinate_grid(offset_cm, r_cm, d, temp, offset_px, first, alpha_deg, res)
    intensity = trilinear_interpolation(X, Y, temp)

    depth = r_px / d

    return intensity, depth, offset_cm


if __name__ == "__main__":
    pass

