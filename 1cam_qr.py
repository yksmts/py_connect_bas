import pypylon.pylon as py
import numpy as np
import cv2
import hook_monitor as hm

# カメラ接続
s_n_1 = "40133253"

inf1 = py.DeviceInfo()
inf1.SetSerialNumber(s_n_1)
cam1 = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice(inf1))
cam1.Open()

# カメラ撮影開始
cam1.ExposureTimeRaw = 10000
cam1.StartGrabbing(py.GrabStrategy_LatestImageOnly)

# 監視クラス
mnt = hm.HookMonitor()

while True:

    # CAM1
    # 画像取得
    grabResult1 = cam1.RetrieveResult(1000)
    if grabResult1.GrabSucceeded():
        img1 = grabResult1.Array
    grabResult1.Release()

    # 表示
    img_rsz1 = cv2.resize(img1, (1000, 1000))
    mnt.get_qr_code(img_rsz1)
    cv2.imshow('CAM1', mnt.qr_img)

    if mnt.new == True:
        mnt.save()
 
    # 終了
    k = cv2.waitKey(1)
    if k == 27:
        break

# 終了処理
cam1.StopGrabbing()
cv2.destroyAllWindows()

cam1.Close()
