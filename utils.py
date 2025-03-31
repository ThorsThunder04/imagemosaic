import os, math
from PIL import Image
import numpy as np
import cv2 as cv

class ImageTile:

    def __init__(self, file_name: str, image_matrix: np.array, image_color: tuple[int,int,int]):
        self.file_name = file_name

        self.size = image_matrix.shape[:2]
        self.image_matrix = image_matrix
        self.image_color = image_color

def name2Num(path):
    """
    @brief renames all files of a given folder to incremental numeric values (while conserving the original extention)
    @param path: string containing the path to the folder"""

    for i, file in enumerate(os.listdir(path)):
        ext = file.split(".")[-1]
        os.rename(os.path.join(path, file), os.path.join(path, str(i) + "." + ext))

def dist3D(p1: tuple[int, int, int], p2: tuple[int, int, int]) -> float:
    """
    Calculates the eucludian distance between two points in 3D space

    Paramaters
    ----------
    p1, p2 : (tuple[int, int, int])
        The two points that we will calculate distance from
    
    Returns
    -------
    (float)
        The distance between the two points
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    sqdist = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    return math.sqrt(sqdist)


def closest_point(pt: tuple[object, tuple],
                  pt_list: list[tuple[object,tuple]],
                  ) -> tuple[object,tuple]:
    """
    Returns the closest point to the target point in 3D space (the returned point isn't the target point)
    
    Parameters
    ----------
    pt : (tuple[object, tuple])
        The target point
    pt_list : (list[tuple[object, tuple]])
        A list of points

    Returns
    -------    
    (tuple[object, tuple])
        The closest point in `pt_list` to `pt`
            
    Notes
    -----
    The target point `pt` and the returned point can both be on the same (x, y, z) position, but they are both distinct points (ex: A(x, y, z) != B(x, y, z)).
    
    This is possible because of the `object` part of a point's type (`str` for example). That represents the point's name.
    """
    target_name, target_coords = pt

    # we need an initial distance, but it can't be 0
    i = 0
    while pt_list[i][0] == target_name: i += 1
    # we store the distance of the point then the point
    closest = (dist3D(target_coords, pt_list[i][1]), pt_list[i])

    for i in range(len(pt_list)):
        if pt_list[i][0] != target_name:
            dist = dist3D(target_coords, pt_list[i][1])

            # if this point is closer then the previous closest point
            if dist < closest[0]:
                closest = (dist, pt_list[i])

    # return the closest point
    return closest[1]


def average_image_color(image_matrix: np.array) -> list[int]:
    """
    Calculates the average of all (r,g,b) values of an image

    Paramaters
    ----------
    image_matrix : (np.array)
        The image we want to calculate the average from
    
    Returns
    -------
    (list[int])
        A list of 3 integers
    """
    return np.mean(image_matrix, (0,1), dtype=np.uint8).tolist()

def loadDirImgs(path: str) -> ImageTile:
    """
    Loads all images of a given directory into a dictionary

    Paramaters
    ----------
    path : (str)
        The directory that contains all the tile images

    Returns
    -------
    (dict[str, ImageTile])
        A dictionary of ImageTile objects where an image filename maps to it's corresponding object
    """

    dict_of_images = {} # will hold dictionary of {filename:image 3D array}
    for file_name in os.listdir(path):
        # load the image into a numpy matrice
        img = cv.imread(os.path.join(path, file_name))
        dict_of_images[file_name] = ImageTile(file_name, img, average_image_color(img))
    
    return dict_of_images

def convert_to_color_space(image_dict: dict[str, dict], conversion_code: int = cv.COLOR_BGR2HSV) -> list[str, np.array]:
    """
    Given a dictionary of images, returns a list of tuples with the image filename and the image converted to a color space

    Paramaters
    ----------
    image_dict : (dict[str, ImageTile])
        A dictionary of ImageTile objects
    conversion_code : (int), default: cv.COLOR_BGR2HSV
        The opencv2 colorspace we wish to convert to
    
    Returns
    -------
    (list[str, np.array])
        A list of tuples. Each tuple contains the image's name and the image's np.array converted to the desired color space
    """

    converted_images = []

    for file_name in image_dict:
        
        # converts the image to the desired color space
        converted_image = cv.cvtColor(image_dict[file_name], conversion_code)

        converted_images.append( (file_name, converted_image) )
    
    return converted_images


if __name__ == "__main__":
    # name2Num("imagestouse")

    target = "./imgs/0.png"

    mockmosaic = mosaicedImage(target, "./imgs")
    print("saving Image!")
    im = Image.fromarray(mockmosaic, "RGB")
    im.save("out.png", "PNG")
