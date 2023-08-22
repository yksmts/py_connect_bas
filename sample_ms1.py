import pypylon.pylon as py
import numpy as np
import cv2

tlf = py.TlFactory.GetInstance()

cam = py.InstantCamera(tlf.CreateFirstDevice())
cam.Open()

# print(cam.PixelFormat.Symbolics)

cam.Height = cam.Height.Max
cam.Width = cam.Width.Max
cam.CenterX = True
cam.CenterY = True

# setup for
cam.PixelFormat = "Mono8"
cam.GainRaw = 136
cam.ExposureTimeRaw = 5000
# print("Resulting framerate:", cam.ResultingFrameRate.Value)

cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
while True:

    # get image
    grabResult = cam.RetrieveResult(2000)
    if grabResult.GrabSucceeded():
        img = grabResult.Array
    grabResult.Release()

    # view
    imag_resize = cv2.resize(img, (1000, 1000))
    cv2.imshow('Linescan View', imag_resize)
    
    # stop
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

# When everything done, release the capture
cam.StopGrabbing()
cv2.destroyAllWindows()

cam.Close()
