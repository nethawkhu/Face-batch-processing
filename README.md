![face](https://github.com/user-attachments/assets/7e41f80e-ea72-4f17-97cf-039518789664)

一个简单的证件人脸照片批处理程序，可将拍摄的照片按分辨率和文件尺寸要求裁剪，人脸自动摆正，人脸宽度占比和垂直位置可调，并可适当处理过曝问题。

处理后的证件照统一放置在 CUT 子目录下。


Pyinstaller 的编译方法：

将 face.py、hook-mediapipe.py 和 1.ico 下载到本地。

输入命令：pyinstaller -F "face.py" -w -i 1.ico --additional-hooks-dir=.
