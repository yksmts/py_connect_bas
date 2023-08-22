import pypylon.pylon as py
import numpy as np
import cv2

# connect
s_n_1 = "40107981"
s_n_2 = "40133253"

inf1 = py.DeviceInfo()
inf1.SetSerialNumber(s_n_1)
cam1 = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice(inf1))
cam1.Open()

inf2 = py.DeviceInfo()
inf2.SetSerialNumber(s_n_2)
cam2 = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice(inf2))
cam2.Open()


# setup for
cam1.ExposureTimeRaw = 10000
cam2.ExposureTimeRaw = 10000

cam1.StartGrabbing(py.GrabStrategy_LatestImageOnly)
cam2.StartGrabbing(py.GrabStrategy_LatestImageOnly)
while True:

    # cam1
    # get image
    grabResult1 = cam1.RetrieveResult(1000)
    if grabResult1.GrabSucceeded():
        img1 = grabResult1.Array
    grabResult1.Release()

    # view
    imag_resize1 = cv2.resize(img1, (1000, 1000))
    cv2.imshow('Linescan View1', imag_resize1)

    # cam2
    # get image
    grabResult2 = cam2.RetrieveResult(1000)
    if grabResult2.GrabSucceeded():
        img2 = grabResult2.Array
    grabResult2.Release()

    # view
    imag_resize2 = cv2.resize(img2, (1000, 1000))
    cv2.imshow('Linescan View2', imag_resize2)
   
    # stop
    k = cv2.waitKey(1)
    if k == 27:
        break

# When everything done, release the capture
cam1.StopGrabbing()
cam2.StopGrabbing()
cv2.destroyAllWindows()

cam1.Close()
cam2.Close()
