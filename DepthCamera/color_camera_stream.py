import depthai as dai
import cv2
import time
import threading


pipeline = dai.Pipeline()
color_cam = pipeline.create(dai.node.ColorCamera)
xout_color = pipeline.create(dai.node.XLinkOut)

class ColorCamera():
    def __init__(self):
        self.color_cam = color_cam
        self.xout_color = xout_color
        self._frame = None
        self._lock = threading.Lock()
        self.pipeline = pipeline
    def set_parameters(self):
        self.color_cam.setPreviewSize(1280, 720)  # set preview
        self.color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.color_cam.setInterleaved(False)  # Choose planar or interleavead representation of data
        self.color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    def link_device(self):
        self.xout_color.setStreamName('ColorCameraOut')
        self.color_cam.preview.link(self.xout_color.input)

        self.device = None
        self.queue = None

    def start_stream(self):
        """Start the pipeline and keep it active until the end of process"""
        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue('ColorCameraOut', maxSize=4, blocking=False)

        while True:
            msg = self.queue.tryGet() #try to get the message in th queue, otherwise returns none
            if msg is not None:
                frame = msg.getCvFrame()
                with self._lock: #lock the process
                    self._frame = frame
            time.sleep(0.01)

    @property
    def frame(self):
        with self._lock:
            if self._frame is None:
                raise ValueError('No frame returned')
            else:
                return self._frame

if __name__ == '__main__':
    color = ColorCamera()
    color.set_parameters()
    color.link_device()
    color.start_stream()
