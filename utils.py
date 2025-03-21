import os, math
from PIL import Image
import numpy as np
import cv2 as cv

class ImageTile:

    def __init__(self, file_name: str, image_matrix: np.array, n_occ: int = 0):
        self.file_name = file_name
        self.n_occ = n_occ # number of occurences of this image tile in the final image

        self.image_matrix = image_matrix
        # the height = rows in matrix, width = columns in matrix
        self.height, self.width = image_matrix.shape[:2]
        self.image_rgb = np.mean(image_matrix, (0,1), dtype=np.uint8)


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
    float
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


def loadDirImgs(path: str):
    """
    Loads all iamges of a given directory into a dictionary

    Paramaters
    ----------
    path : str
        The directory that contains all the tile images

    Returns
    -------
    (dict[str, np.array])
        A dictionary of all the loaded images mapped to their file names
    """

    dict_of_images = {} # will hold dictionary of {filename:image 3D array}
    for img in os.listdir(path):
        dict_of_images[img] = cv.imread(os.path.join(path, img))
    
    return dict_of_images

def cvtHSV(imgsDict: dict) -> dict:
    """
    @brief takes in a dictionary of images and returns a copy of the dictionary, with all images converted from RGB to HSV

    @param imgsDict: dictionary of numpy image arrays
    @returns: dictionary of HSV images
    """

    hsvDict = {}

    for key, value in imgsDict.items():
        hsvDict[key] = cv.cvtColor(value, cv.COLOR_BGR2HSV)

    return hsvDict

def avgNpImgMat(npMat) -> list[float]:
    """
    @brief given a 3D numpy array of an image, calculates the average rgb value of the pixels
    
    @param npMat: 3D numpy array of an image
    @returns: list containing the average rgb values
    """
    pixels = npMat.shape[0]*npMat.shape[1] # calculates how many pixels there are
    avgVals = npMat.sum(axis=0).sum(axis=0) # sums all the rgb values together
    avgVals = avgVals / pixels # divide each rgb value by the total number of pixels
    return avgVals.tolist()

def pythonicAvg(matr): # does same as avgNpImgMat but with just python
    pmat = matr.tolist()
    pixels = len(pmat)* len(pmat[0])
    s = [0]*3
    for a in pmat:
        for b in a:
            for i in range(len(b)):
                s[i] += b[i]
    return [x/pixels for x in s]

def imgFromData(imgsDir, imgData):
    """
    @brief Given the calculated data taht contains the info needed to know what images to place where, we create the new image that will contain all of the tiles
    
    @param imgsDir: directory containing tile images
    @param imgData: matrix containing info on what images to place on what pixels. Form: `[[(imgtitle, avgRGBOfImg, dist2targetPx), ...], ...]`
    @returns: numpy array matrix containing image rgb data (values in uint8)
    """

    #TODO optimize space by only loading an image when it's needed.
    #TODO loading all images at once is not necessary for most of them might not even be used
    images = {}
    for image in os.listdir(imgsDir):
        images[image] = cv.imread(os.path.join(imgsDir, image))
    
    #TODO UN-hardcode the 256 to allow for different in and out resolution
    # newImg = Image.new("RGB", (256**2, 256**2))
    newImg = np.zeros((256**2, 256**2, 3), dtype="uint8")
    
    for i in range(len(imgData)):
        for j in range(len(imgData[0])):
            # print(i, j)
            newImg[256*i:256*(i+1), 256*j:256*(j+1)] = images[imgData[i][j][0]]
            # newImg.paste(images[imgData[i][j][0]], (256*j, 256*i))
    
    return newImg


def mosaicedImage(targetImage, imagesDir, reuseImages=True):
    #TODO Major rewrite to allow more flexibility in the settings

    timg = cv.imread(targetImage) # matr of target image
    mosImgs = loadDirImgs(imagesDir) 
    hsvMosImgs = cvtHSV(mosImgs)
    avgHSVImgs = [(name, avgNpImgMat(matr)) for name, matr in hsvMosImgs.items()]
    print(len(avgHSVImgs))

    mockMosaic = []
    for i in range(timg.shape[0]):
        mockMosaic.append([])
        for j in range(timg.shape[1]):
            mockMosaic[i].append(0)



    hsvtimgList = cvtHSV({targetImage: timg})[targetImage].tolist()

    for i in range(len(mockMosaic)):
        for j in range(len(mockMosaic[0])):
            ttl, px, dist = kppv((targetImage.split("/")[-1], hsvtimgList[i][j]), avgHSVImgs, 1)[0]
            px = [math.floor(n) for n in px]
            mockMosaic[i][j] = (ttl, px, dist)
    
    newimg = imgFromData(imagesDir, mockMosaic)
    return newimg





if __name__ == "__main__":
    # name2Num("imagestouse")

    target = "./imgs/0.png"

    mockmosaic = mosaicedImage(target, "./imgs")
    print("saving Image!")
    im = Image.fromarray(mockmosaic, "RGB")
    im.save("out.png", "PNG")
