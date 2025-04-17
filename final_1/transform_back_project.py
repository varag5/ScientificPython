import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def read_png(path):
    img = plt.imread(path)
    return img 

def read_png_colored(path):
    img = plt.imread(path)
    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    return img_gray


class VolumeTransformer:
    def __init__(self, frame_width: int, frame_depth: int, depth: float, thetas: Tuple[float, float],offset: float, resolution: float, straighten_volume=False):
        """
        Initializes the TransformVolume object.

        Args:
            frame_width (int): The width of the frame.
            frame_height (int): The height of the frame.
            frame_depth (int): The depth of the frame.
            depth (float): Depth of the frustum.
            thetas (Tuple[float, float]): The range of theta values.
            phis (Tuple[float, float]): The range of phi values.
            offset (float): The offset value between the top of the frame and the US source.
            resolution (float): The resolution value that could be used to upscale or downscale the output volume.
            straighten_volume (bool, optional): If this is set to True, the volume will be straightened. Defaults to False.
        """
        self.depth = depth
        self.thetas = thetas
        if straighten_volume:
            self.thetas = np.array([-abs(thetas[1]-thetas[0])/2, abs(thetas[1]-thetas[0])/2])
        self.offset = offset
        self.resolution = resolution

        self.frame_width, self.frame_depth = frame_width, frame_depth
        self.frame_THETA = np.linspace(self.thetas[0], self.thetas[1], self.frame_width)
        #self.frame_depth = 
        self.frame_R = np.linspace(offset, depth+offset, self.frame_depth)

        self.create_image_volume()
        self.find_nearest()

    def transform_spherical_to_cartesian(self,R: np.ndarray, THETA: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms spherical coordinates to Cartesian coordinates.

        Args:
            R (np.ndarray): Array of radial distances. 
            THETA (np.ndarray): Array of polar angles.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the X, Y, and Z coordinates in Cartesian space.

        All parameters and return values have dr x dtheta x dphi dimensions 
        """
            
        Z = np.cos(THETA)*R
        X = np.sin(THETA)*R

        return X, Z

    def transform_cartesian_to_spherical(self, X: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms Cartesian coordinates to spherical coordinates.

        Parameters:
        X (np.ndarray): Array of X coordinates.
        Y (np.ndarray): Array of Y coordinates.
        Z (np.ndarray): Array of Z coordinates.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the transformed R, THETA, and PHI coordinates.

        All parameters and return values have dr x dtheta x dphi dimensions 
        """

        R = np.sqrt(X**2  + Z**2)
        THETA = np.zeros(R.shape)
        THETA[R>0] = np.arcsin(X[R>0]/R[R>0])

        return R, THETA

    def create_image_volume(self):
        """
        Create the 3D cartesian (X,Y,Z) and equivalent spherical coordinates of the image volume, and a mask for the cone of interest
        """

        R, THETA = np.meshgrid(self.frame_R, self.frame_THETA,  indexing='ij')
        X, Z = self.transform_spherical_to_cartesian(R, THETA)
        x_min = np.min(X)
        x_max = np.max(X)
        z_max = np.max(Z)

        x_len = int(np.ceil((x_max-x_min)/self.resolution+1))
        z_len = int(np.ceil(z_max/self.resolution+1))

        X = np.linspace(x_min, x_max, x_len)
        X = np.tile(X, (z_len, 1)).T
        Z = np.linspace(0, z_max, z_len)
        Z = np.tile(Z, (x_len, 1))

        R, THETA = self.transform_cartesian_to_spherical(X,Z)

        THETA_small = (THETA < np.min(self.thetas))#-0.05)
        THETA_large = (THETA > np.max(self.thetas))#+0.025)
        R_large = (R > self.depth+self.offset)

        cone_mask = (THETA_small | THETA_large | R_large)

        self.image_shape = X.shape
        self.image_R = R[cone_mask == False]
        self.image_THETA = THETA[cone_mask == False]
        self.cone_mask_indices = np.where(cone_mask == False)
        
    def find_nearest(self):
        """
        Find the 4 nearest indices and values for the trilinear interpolation, and assing a weight to each value
        """

        dR = np.mean(np.diff(self.frame_R))
        dTHETA = np.mean(np.diff(self.frame_THETA))
        
        THETA_l_ind = np.floor((self.image_THETA - self.frame_THETA[0])/dTHETA).astype(int)
        THETA_l_ind[THETA_l_ind < 0] = 0
        THETA_l_ind[THETA_l_ind >= self.frame_width-1] = self.frame_width-2
        THETA_l_val = self.frame_THETA[THETA_l_ind]
        THETA_r_ind = THETA_l_ind + 1
        THETA_r_val = self.frame_THETA[THETA_r_ind]

        R_l_ind = np.floor((self.image_R - self.frame_R[0])/dR).astype(int)
        R_l_ind[R_l_ind < 0] = 0
        R_l_ind[R_l_ind >= self.frame_depth-1] = self.frame_depth-2
        R_l_val = self.frame_R[R_l_ind]
        R_r_ind = R_l_ind + 1
        R_r_val = self.frame_R[R_r_ind]

        left_Theta = self.image_THETA - THETA_l_val
        right_Theta = THETA_r_val - self.image_THETA
        R_m = -(2*self.image_R*(np.cos(right_Theta)-np.cos(left_Theta))) / (np.power(2*np.sin((THETA_r_val - THETA_l_val)/2),2)*(np.sin(right_Theta) - np.sin(left_Theta))/(np.sin(right_Theta) + np.sin(left_Theta)))
        R_m[np.cos(right_Theta)==np.cos(left_Theta)] = self.image_R[np.cos(right_Theta)==np.cos(left_Theta)]/np.cos(dTHETA/2)
        self.THETA_l_ind = THETA_l_ind
        self.THETA_r_ind = THETA_r_ind
        self.R_l_ind = R_l_ind
        self.R_r_ind = R_r_ind

        self.w1 = (R_r_val - R_m)/(R_r_val - R_l_val)
        self.w2 = 1 - self.w1
        w3_a = np.sin(right_Theta)
        w3_b = np.sin(left_Theta)
        self.w3 = w3_a/(w3_a + w3_b)
        self.w4 = 1 - self.w3


    def trilinear_interpolation(self,frame: np.ndarray) -> np.ndarray:
        """
        Perform trilinear interpolation on the input frame using the 8 neighrest indices and the corresponding weights.
        """

        A = frame[self.THETA_l_ind,  self.R_l_ind]
        B = frame[self.THETA_r_ind,  self.R_l_ind]
        C = frame[self.THETA_r_ind,  self.R_l_ind]
        D = frame[self.THETA_r_ind,  self.R_r_ind]

        interpolated_frame = np.zeros(self.image_shape)
        to_fill = self.w1*(self.w3*A + self.w4*B) + self.w2*(self.w3*C + self.w4*D)
        interpolated_frame[self.cone_mask_indices] = to_fill

        return interpolated_frame




def interp(path, depth, alpha, offset_frame, offset=0, resolution=0.01):
    """
    Interpolates the image using the VolumeTransformer class.

    Args:
        path (str): Path to the image file.
        depth (float): Depth of the frustum.
        thetas (Tuple[float, float]): The range of theta values.
        alpha (float): Half angle of the ultrasound device in radians.
        offset (float): Offset value between the top of the frame and the US source.
        offset_frame (float): Offset value for the frame.
        resolution (float): Resolution value for upscaling or downscaling.

    Returns:
        np.ndarray: Interpolated frame.
    """
    thetas = (-alpha, alpha)
    offset = 0

    # Read and process the image
    frame = read_png_colored(path).T
    frame_width, frame_depth = frame.shape

    # Create padded frame
    offset_row = int(frame_depth / depth * offset_frame)
    padded_frame = np.zeros((frame_width, offset_row + frame_depth))
    padded_frame[:, offset_row:] = frame[:, :]

    # Update depth
    frame = padded_frame
    frame_depth = frame.shape[1]  # must match actual depth now

    # Pass correct dimensions
    transformer = VolumeTransformer(frame_width, frame_depth, depth, thetas, offset, resolution, straighten_volume=True)

    interpolated_frame = transformer.trilinear_interpolation(frame.astype(np.float32))
    
    return interpolated_frame.T







def interp_img(frame, depth, alpha, offset_frame, offset=0, resolution=0.01):
    """
    Interpolates the image using the VolumeTransformer class.

    Args:
        path (str): Path to the image file.
        depth (float): Depth of the frustum.
        thetas (Tuple[float, float]): The range of theta values.
        alpha (float): Half angle of the ultrasound device in radians.
        offset (float): Offset value between the top of the frame and the US source.
        offset_frame (float): Offset value for the frame.
        resolution (float): Resolution value for upscaling or downscaling.

    Returns:
        np.ndarray: Interpolated frame.
    """
    thetas = (-alpha, alpha)
    offset = 0

    # Read and process the image
    #frame = read_png_colored(path).T
    frame = frame.T
    frame_width, frame_depth = frame.shape

    # Create padded frame
    offset_row = int(frame_depth / depth * offset_frame)
    padded_frame = np.zeros((frame_width, offset_row + frame_depth))
    padded_frame[:, offset_row:] = frame[:, :]

    # Update depth
    frame = padded_frame
    frame_depth = frame.shape[1]  # must match actual depth now

    # Pass correct dimensions
    transformer = VolumeTransformer(frame_width, frame_depth, depth, thetas, offset, resolution, straighten_volume=True)

    interpolated_frame = transformer.trilinear_interpolation(frame.astype(np.float32))
    
    return interpolated_frame.T





if __name__ == "__main__":
    pass
    '''depth = 8.65#6
    thetas = (-np.pi/3, np.pi/3)
    #alpha = 84/180/2*np.pi
    alpha = 35/180*np.pi
    thetas = (-alpha, alpha)
    offset = 0#1.37 #0
    offset_frame = 1.37
    resolution = 0.01
    path = "C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\Korea\\SouthKorea2025\\data\\masked_image_Marcitol_kapott_egyenes_2.png"#masked_image.png"

    frame = interp(path, depth, thetas, alpha, offset, offset_frame, resolution)

    plt.imshow(frame, cmap="gray")
    plt.show()



'''