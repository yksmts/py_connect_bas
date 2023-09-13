import cv2
from hook_monitor import HookMonitor

img = cv2.imread("2.png", cv2.IMREAD_GRAYSCALE)
hk = HookMonitor()
hk.get_qr_code(img)
print(hk.qr_code)
