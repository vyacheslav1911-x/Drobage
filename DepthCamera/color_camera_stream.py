import depthai as dai
import cv2
import numpy as np
import threading

pipeline = dai.Pipeline()
color_cam = pipeline.create(dai.node.ColorCamera)
xout_color = pipeline.create(dai.node.XLinkOut)

class ColorCamera():
    def __init__(self):
        self.color_cam = color_cam
        self.xout_color = xout_color
        self._frame = None

    def set_parameters(self):
        self.color_cam.setPreviewSize(1280, 720)  # set preview
        self.color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.color_cam.setInterleaved(False)  # Choose planar or interleavead representation of data
        self.color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    def link_device(self):
        self.xout_color.setStreamName('ColorCameraOut')
        self.color_cam.preview.link(self.xout_color.input)


    def start_stream(self):
        with dai.Device(pipeline) as device:
            previewQueue_ColorCamera = device.getOutputQueue('ColorCameraOut', maxSize=4, blocking=True)
            def stream_color_camera():
                while True:
                    try:
                        if previewQueue_ColorCamera:
                            frame_color = previewQueue_ColorCamera.get() #get the next frame from the queue
                            self._frame = frame_color.getCvFrame() #conversion into NumPy array suitable for further image processing
                            cv2.imshow('ColorCameraOut', self._frame)
                    except:
                        print('ColorCamera queue is empty')
                        break;

                    if cv2.waitKey(1) == ord('q'):
                        break;

                cv2.destroyAllWindows()

            stream_color_camera()

    @property
    def frame(self):
        if self._frame is None:
            raise ValueError('No frame returned')
        else:
            return self._frame

if __name__ == '__main__':
    color = ColorCamera()
    color.set_parameters()
    color.link_device()
    color.start_stream()
