from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import math as m

# In medical ultrasound imaging, there are transducers that create a curved image
# We use this code to transform the curved image to a flat image
# We have to get the polar coordinates of the image, then we can transform it to a cartesian coordinate system

# The image we have is 


# Load the image
image_path = "C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\transforming recordings\\samples\\sample_curved_cropped_01.png"  # Replace with your image file path
image = Image.open(image_path)

# Convert to grayscale
gray_image = image.convert("L")  # "L" mode is for grayscale

# Convert to NumPy array
gray_array = np.array(gray_image)

# We are searching for white marks on the image that are 1 cm apart
# We are looking for the first pixels where the intensity gets above 10, that would be the start of each cm mark
# We add the index of the first pixel of each mark to the list
# First we do this for the columns, then for the rows
a = b = False
column = []
for i in range(1230):
    if gray_array[2][i] > 10:
        a = True
    else:
        a = False
    if a != b and a == True:
        column.append(i)
    b = a
print('col ', column)

a = b = False
row = []
for i in range(790):
    if gray_array[i][2] > 10:
        a = True
    else:
        a = False
    if a != b and a == True:
        row.append(i)
    b = a
print('row ', row)

# To get the average distance between the marks we calculate the distance between the first and the last cm mark and divide it by the number of cm marks minus 1
row_cm = (row[-1]-row[1]) / (len(row)-2)
column_cm = (column[-1]-column[1]) / (len(column)-2)
height = row_cm
width = column_cm

# We want to clear all the noise from the image
# To get the row where the marks end and the actual image starts, we are looking for the first row where the sum of the intensity is lower than 1500
gray_row_sums = np.sum(gray_array, axis = 1)
gray_row_id = np.where(gray_row_sums<1500)[0][0]

# We copy the part of the image where there are no marks and then we convert the image to float
# We set the values above 200 and below 5 to 0 removing too bright and too dark spots that are not from the ultrasound device
# We set the values in the top left corner and the top right corner to 0 to remove other marks which are still visible
temp = gray_array[gray_row_id:,6:].copy().astype(float)
temp[temp>200] = 0
temp[temp<5] = 0
temp[:100,:100] = 0
temp[:100,-200:] = 0
row_sums = np.sum(temp, axis = 1)



# The transducer records data in a given angle. There will be data outside a certain radius (offset).
# To get the angle of the ultrasound device, we need the points called column ids - two vertices located on the arc/line of the radius.
# After getting thoses values, we can generate a triangle to calculate the angle.
#  Theoratically, all the three traingle's edges are determined with the help of the two column id points. The angle can be calcualted with the help of the edges.
# Additionally the radius could be derived from those calculations (d/sin(theta/2)). It could be used as a parameter for the polar coordinates.

# In this particular setup the angle of the ultrasound device is 84 degrees - calculated in a different environment.
theta = 84/180*np.pi
alpha_image = theta/2



# We are looking for the two column ids of the image.
row_id = np.where(row_sums>200)[0][0]
column_ids = sc.signal.find_peaks(temp[row_id,:],distance=100)[0]
cln = column_ids.copy()
for i in range(len(column_ids)):
    if abs(column_ids[i]) > len(gray_array[0])*2/3:
        cln = np.delete(column_ids, i)    

column_sums = np.sum(temp, axis = 0)

d = abs(cln[0] - cln[1])/2
d_cm = d/column_cm



# The image is distorted because one centimeter contains more pixels horizontally than vertically. We have to use this ratio to get the real angle of the image.
ratio = height / width
alpha_real = m.atan(m.tan(alpha_image)*ratio)


offset_cm = d_cm/np.sin(alpha_real)
offset = offset_cm*height 

middle_column = m.floor(cln[0] + d)


# We are looking for the first and last nonzero element of the middle column, getting the radius of the actual image - that is without the offset radius.
middle_mask = np.where(temp[:, middle_column] > 0)
first = middle_mask[0][0]
last = middle_mask[0][-1]

r = last - first

# We perform unit conversions for later calculations
r_cm = r/height

r_mm = r_cm*10
offset_mm = offset_cm*10

alpha_deg = alpha_real*180/m.pi
alpha = round(alpha_deg)

# We calculate the meshgrid for the transformation
# The R values of the polar coordinates range from the offset radius to offset radius + radius of the image
# The radius for the polar coordinates will strart from the offset radius value and go to the offset + radius value - radius is the height of the actual image.
# The range of the angle is from -alpha to alpha - alpha is half the angle of the ultrasound device. It is easier to calculate the transformation this way that using only positive values.
# Th_d and R are the meshgrid containing the polar coordinates for each point

rows = np.arange(m.floor(offset_mm), m.floor(offset_mm + r_mm))  # 23 to 788 inclusive
cols = np.arange(-alpha, alpha)   # 27 to 81 inclusive

Th_d, R = np.meshgrid(cols, rows)


Th = Th_d/180*m.pi

# X and Y are the cartesian coordinates, which we calculate from the polar coordinates.
Y = np.cos(Th)*R
X = np.sin(Th)*R


x_pixel = X/10*width
x_pixel += temp.shape[1]/2

y_pixel = Y/10*height-(offset-first)


# We calculate the intensity of x, y in the transformed space based on the 4 adjascent pixels intensity.
# The intensity will be like a weighted sum.


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
            Intensity[i][j] = (((x-xl)*C)/(xr-xl) + ((xr-x)*A)/(xr-xl)*(yb-y))/(yb-yt) + (((x-xl)*D)/(xr-xl) + ((xr-x)*B)/(xr-xl)*(y-yt))/(yb-yt)
            if Intensity[i][j] < 0:
                Intensity[i][j] = 0
            if Intensity[i][j] > 255:
                Intensity[i][j] = 255
            #Intensity[i][j] = 255 - Intensity[i][j]
    Intensity = np.clip(Intensity, 0, 255)

    return Intensity

# Showing the result of the instensity
Intensity = trilinear_interpolation(x_pixel, y_pixel, temp)
print('intensity ', Intensity)
plt.imshow(Intensity)
plt.show()

# Saving the image
import PIL
img = PIL.Image.fromarray(np.uint8(Intensity))
img.save("C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\Korea\\SouthKorea2025\\transformation_curved_flat_whole_1_test.png")
