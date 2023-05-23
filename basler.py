from pypylon import pylon
import cv2
import os,stat

script_dir =  os.path.abspath(os.path.dirname(__file__))
def capture_image(image_name,exp,gain_value):

    tl_factory = pylon.TlFactory.GetInstance()

    # 獲取所有已連接的相機
    devices = tl_factory.EnumerateDevices()

    if len(devices) == 0:
        print("未檢測到相機")
        return

    # 選擇第一個相機
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

    # 開啟相機
    camera.Open()

    try:
        # 設置相機參數
        camera.PixelFormat = "Mono8"  # 可以根據需要選擇不同的像素格式
        camera.ExposureTime = exp  # 曝光時間（微秒）
        camera.Gain = gain_value
        # 拍攝一張照片
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            # 將照片轉換為NumPy數組
            image = grab_result.Array
            os.chmod(script_dir,stat.S_IWRITE)
            cv2.imwrite(os.path.join(script_dir,image_name),image)
            print("Save image:{}".format(os.path.join(script_dir,image_name)))

            # 顯示照片寬度和高度
            print("Image width:", image.shape[1])
            print("Image height:", image.shape[0])

        grab_result.Release()
    finally:
        # 關閉相機
        camera.Close()


# 執行拍照函數
if __name__=='__main__':
    capture_image(r"img_ori_1p.png",8346*4,0)