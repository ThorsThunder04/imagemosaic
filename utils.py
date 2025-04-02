import os, math
import numpy as np
import cv2 as cv
import threading as thd

class ImageTile:

    def __init__(self, file_name: str, image_matrix: np.ndarray, image_color: tuple[int,int,int]):
        self.file_name = file_name

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


def average_image_color(image_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the average of all (r,g,b) values of an image

    Paramaters
    ----------
    image_matrix : (np.ndarray)
        The image we want to calculate the average from
    
    Returns
    -------
    (np.ndarray)
        A list of 3 integers
    """
    return np.floor(np.mean(image_matrix, (0,1)))

def load_imgs_from_dir(path: str) -> dict[str, ImageTile]:
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

def convert_images_to_color_space(image_dict: dict[str, ImageTile], conversion_code: int = cv.COLOR_BGR2HSV) -> list[str, np.ndarray]:
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
    (list[str, np.ndarray])
        A list of tuples. Each tuple contains the image's name and the image's np.ndarray converted to the desired color space
    """

    converted_images = []

    for file_name, tile in image_dict.items():
        
        # converts the image to the desired color space
        converted_image = cv.cvtColor(tile.image_matrix, conversion_code)

        converted_images.append( (file_name, converted_image) )
    
    return converted_images

def calculate_image_positions(target_filename: str, target_image: np.ndarray, tile_images: dict[str, ImageTile]) -> tuple[list[list[str]], set[str]] :
    """
    Calculates which images to place on which pixels
    The positions will be represented by a 2d list of the same dimensions as `target_image`. 

    Paramaters
    ----------
    target_filename : (str)
        The filename of the target image
    target_image : (np.ndarray)
        The target image for the mosaic
    tile_images : (dict[str, ImageTile])
        The images we will use a tiles for the mosaic
    
    Returns
    -------
    (list[list[str]])
        List of same dimentions as `target_image` containing an image name in the place of each pixel
    (set[str])
        A set of all images that are used in the resulting mosaic (will be used to free up memory)
    """

    target_height, target_width = target_image.shape[:2]
    image_positions = [[""]*target_width for _ in range(target_height)]
    # print(image_positions)
    used_images = set()

    # we will used HSV to determin how close an image is to another in 3d space
    #NOTE This method should probably be revisited to get a more accurate idea of "closeness".
    print("Converting to HSV")
    converted_image_averages = [(title, average_image_color(img)) for title,img in convert_images_to_color_space(tile_images, cv.COLOR_BGR2YUV)]
    converted_target = cv.cvtColor(target_image, code = cv.COLOR_BGR2YUV)

    print("Finding closest image for each pixel")
    # start calculating the positions
    for r in range(target_height):
        for c in range(target_width):

            # calculates the image with the color that resembles the pixel color at (r,c) the most
            closest_image = closest_point((target_filename, converted_target[r, c]), converted_image_averages)

            used_images.add(closest_image[0])
            image_positions[r][c] = closest_image[0]
    
    return image_positions, used_images

def place_image_at(row: int,
                   row_img_names: list[str],
                   result_image: np.ndarray,
                   tile_images: dict[str, ImageTile],
                   tile_size: int
                   ) -> None:
    """
    Given a row to start at, place each image from `row_img_names` on it's respective position in result_image

    Paramaters
    ----------
    row : (int)
        The row of pixels to start placing images for
    row_img_names : (list[str])
        The respective image filenames for each pixel in that row
    result_image : (np.ndarray)
        Refernce to the image we are placing all the tile images onto
    tile_images : (dict[str, ImageTile])
        All the tile time images that will be used for the result image
    tile_size : (int)
        The 1:1 size of each tile image
    """

    # for each pixel on the given row
    for c in range(len(row_img_names)):
        # get the tile image's matrice
        image_to_place = tile_images[row_img_names[c]].image_matrix


        result_image[tile_size*row:tile_size*(row+1), tile_size*c:tile_size*(c+1)] = image_to_place
        # result_image[row:c] = image_to_place
    
    
    

def create_mosaic(target_filename: str, 
                  tile_images_dir: str,
                  tile_size: int = 64,
                  target_image_size: (tuple[int, int]|None) = None
                  ) -> None:
    """
    From a source image, create the mosaiced image with tile images

    Paramaters
    ----------
    target_filename : (str)
        The path to source image that the smaller tile images will try to recreate
    tile_images_dir : (str)
        The directory path to all the tile images
    tile_size : (int), default: 64
        The size that each tile image should be. Each tile image will be resized to an 1:1 ratio of this size
    target_image_size : (tuple[int,int]|None), default: None
        A tuple for the resolution of the target image, of the form (WIDTH, HEIGHT)
        If `None` is given, then it will take the original resolution of the target image.
    """


    # initial load of data
    target_image = cv.imread(target_filename)
    # if a custom target size is given
    if target_image_size is not None:
        target_w, target_h = target_image_size
        target_image = cv.resize(target_image, (target_w, target_h))
    else:
        target_h, target_w = target_image.shape[:2]
    tile_images = load_imgs_from_dir(tile_images_dir)

        

    # resize tile images to `tile_size`
    for _, img in tile_images.items():
        img.image_matrix = cv.resize(img.image_matrix, (tile_size, tile_size))
    print("Resize Done")
    

    # calculate where to place each image, and also save what images will be used
    mosaic_plan, used_images = calculate_image_positions(target_filename, target_image, tile_images)
    print("Positions Calculated")
    # print(used_images)
    
    # clear unused images
    img_keys = list(tile_images.keys())
    for key in img_keys:
        if key not in used_images:
            del tile_images[key]


    # creates an empty image matrix for the result
    result_image = np.zeros((tile_size*target_w, tile_size*target_h, 3), dtype=np.uint8)

    print("Placing tiles")
    # place all the tile images for each row of the target image
    for row in range(len(mosaic_plan)):

        place_image_at(row,
                       mosaic_plan[row],
                       result_image,
                       tile_images,
                       tile_size)
        # print(row, target_h)
    
    return result_image
    
    

if __name__ == "__main__":


    result_image = create_mosaic("./imgs/1.png", "./us", 32, (64,64)) 
    cv.imwrite("out.png", result_image)
