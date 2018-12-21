import cv2

img = cv2.imread('target.png', 0)
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:  # 等待ESC直接退出
    cv2.destroyAllWindows()
elif k == ord('s'):  # 等待输入s键，然后复制文件再退出
    cv2.imwrite('target_1.png', img)
    cv2.destroyAllWindows()
