import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from enum import Enum

class DrawOption(Enum):
    AXES = 1
    CUBE = 2

def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)

    corner = tupleOfInts(corners[0].ravel())
    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Add green plane
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Add box borders
    for i in range(4):
        j = i + 4
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 3)
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

def poseEstimation(option: DrawOption):
    
    root = os.getcwd()
    paramPath = os.path.join(root, 'calibration.npz')
    data = np.load(paramPath)
    camMatrix = data['camMatrix']
    distCoeff = data['distCoeff']

    # Read image
    calibrationDir = os.path.join(root, '20')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    nRows, nCols = 19, 19
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    worldPtsCur = np.zeros((nCols * nRows, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

    # World points of objects to be drawn
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
    cubeCorners = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                              [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    # Find corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgBGR = cv.resize(imgBGR, (600, 600))
        if imgBGR is None:
            print(f"Warning: Could not read image at path {curImgPath}")
            continue

        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound:
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            success, rvecs, tvecs = cv.solvePnP(worldPtsCur, cornersRefined, camMatrix, distCoeff)

            if option == DrawOption.AXES:
                imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawAxes(imgBGR, cornersRefined, imgpts)

            if option == DrawOption.CUBE:
                imgpts, _ = cv.projectPoints(cubeCorners, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawCube(imgBGR, imgpts)

            cv.imshow('Chessboard', imgBGR)
            cv.waitKey(1000)

if __name__ == '__main__':
    #poseEstimation(DrawOption.AXES)
    poseEstimation(DrawOption.CUBE)
    
