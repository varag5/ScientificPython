import numpy as np
from PIL import Image
import scipy.signal
import math as m

import os
import math as m
import numpy as np
from PIL import Image
import scipy.signal
import matplotlib.pyplot as plt


def load_image(image_path):
    cfd = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(cfd, "data", "nevtelen2.png")
    print("Loaded image from:", image_path)
    return Image.open(image_path), image_path


def convert_to_grayscale(image):
    return np.array(image.convert("L"))


def detect_cm_marks(gray_array):
    column = []
    a = b = False
    for i in range(gray_array.shape[1]):
        a = gray_array[2][i] > 10
        if a != b and a:
            column.append(i)
        b = a

    row = []
    a = b = False
    for i in range(gray_array.shape[0]):
        a = gray_array[i][2] > 10
        if a != b and a:
            row.append(i)
        b = a

    print('Row marks:', row)
    print('Column marks:', column)
    return row, column


def clean_image(gray_array):
    gray_row_sums = np.sum(gray_array, axis=1)
    gray_row_id = np.where(gray_row_sums < 1500)[0][0]

    temp = gray_array[gray_row_id:, 6:].copy().astype(float)
    temp[temp > 200] = 0
    temp[temp < 5] = 0
    temp[:100, :100] = 0
    temp[:100, -200:] = 0
    return temp, gray_row_id


def calculate_geometry(temp, row, column, height, width):
    row_sums = np.sum(temp, axis=1)
    row_id = np.where(row_sums > 200)[0][0]
    column_ids = scipy.signal.find_peaks(temp[row_id, :], distance=100)[0]

    cln = column_ids.copy()
    for i in range(len(column_ids)):
        if abs(column_ids[i]) > len(temp[0]) * 2 / 3:
            cln = np.delete(column_ids, i)

    d = abs(cln[0] - cln[1]) / 2
    d_cm = d / width
    theta = 84 / 180 * m.pi
    alpha_image = theta / 2
    ratio = height / width
    alpha_real = m.atan(m.tan(alpha_image) * ratio)
    offset_cm = d_cm / m.sin(alpha_real)
    offset = offset_cm * height
    middle_column = m.floor(cln[0] + d)

    middle_mask = np.where(temp[:, middle_column] > 0)
    first = middle_mask[0][0]
    last = middle_mask[0][-1]
    r = last - first

    return offset, r, offset_cm, r / height, alpha_real, first


def compute_coordinate_grid(offset_cm, r_cm, alpha_real, height, width, temp_shape, offset_pixel, first):
    r_mm = r_cm * 10
    offset_mm = offset_cm * 10
    alpha_deg = alpha_real * 180 / m.pi
    alpha = round(alpha_deg)

    rows = np.arange(m.floor(offset_mm), m.floor(offset_mm + r_mm))
    cols = np.arange(-alpha, alpha)

    Th_d, R = np.meshgrid(cols, rows)
    Th = Th_d / 180 * m.pi

    Y = np.cos(Th) * R
    X = np.sin(Th) * R

    x_pixel = X / 10 * width + temp_shape[1] / 2
    y_pixel = Y / 10 * height - (offset_pixel - first)
    return x_pixel, y_pixel


def trilinear_interpolation(X, Y, temp):
    X_left = np.floor(X).astype(int)
    X_right = np.ceil(X).astype(int)
    Y_top = np.floor(Y).astype(int)
    Y_bottom = np.ceil(Y).astype(int)

    Intensity = np.zeros(X.shape)
    for i in range(len(Intensity)):
        for j in range(len(Intensity[0])):
            x = X[i][j]
            y = Y[i][j]
            xl = X_left[i][j]
            xr = X_right[i][j]
            yt = Y_top[i][j]
            yb = Y_bottom[i][j]
            A = temp[yt, xl]
            B = temp[yb, xl]
            C = temp[yt, xr]
            D = temp[yb, xr]
            Intensity[i][j] = (((x - xl) * C) / (xr - xl) + ((xr - x) * A) / (xr - xl) * (yb - y)) / (yb - yt) + \
                              (((x - xl) * D) / (xr - xl) + ((xr - x) * B) / (xr - xl) * (y - yt)) / (yb - yt)
            Intensity[i][j] = np.clip(Intensity[i][j], 0, 255)
    return Intensity


def save_image(intensity_array, output_path):
    img = Image.fromarray(np.uint8(intensity_array))
    img.save(output_path)
    print("Saved image to:", output_path)


def main():
    image, _ = load_image()
    gray_array = convert_to_grayscale(image)
    row, column = detect_cm_marks(gray_array)

    row_cm = (row[-1] - row[1]) / (len(row) - 2)
    column_cm = (column[-1] - column[1]) / (len(column) - 2)

    height = row_cm
    width = column_cm

    temp, gray_row_id = clean_image(gray_array)

    offset_pixel, r_pixel, offset_cm, r_cm, alpha_real, first = calculate_geometry(temp, row, column, height, width)

    x_pixel, y_pixel = compute_coordinate_grid(offset_cm, r_cm, alpha_real, height, width, temp.shape,
                                               offset_pixel, first)

    intensity = trilinear_interpolation(x_pixel, y_pixel, temp)

    plt.imshow(intensity)
    plt.show()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "transformed_image.png")
    save_image(intensity, save_path)


if __name__ == "__main__":
    main()
    print("Done")
    print("Program finished successfully.")
    print("All tasks completed.")
    print("Exiting program.")
    print("Thank you for using the program.")
    print("Goodbye!")
    print("Have a nice day!")
    print("See you next time!")
    print("Take care!")
    print("End of program.")
def transform_curved_to_flat(image_path):
    image = Image.open(image_path).convert('L')
    gray_array = np.array(image)
    row, column = detect_cm_marks(gray_array)
    height = (row[-1]-row[1])/(len(row)-2)
    width = (column[-1]-column[1])/(len(column)-2)
    temp, gray_row_id = clean_image(gray_array)
    offset_pixel, r_pixel, offset_cm, r_cm, alpha_real, first = calculate_geometry(temp, row, column, height, width)
    x_pixel, y_pixel = compute_coordinate_grid(offset_cm, r_cm, alpha_real, height, width, temp.shape, offset_pixel, first)
    intensity = trilinear_interpolation(x_pixel, y_pixel, temp)
    return intensity
