import depthai as dai
import cv2
import numpy as np

pipeline = dai.Pipeline()
color_cam = pipeline.create(dai.node.ColorCamera)  # create color camera node
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo_cam = pipeline.create(dai.node.StereoDepth)
xout_stereo = pipeline.create(dai.node.XLinkOut)  # create xlink output node
xout_color = pipeline.create(dai.node.XLinkOut)


class StereoCamera():
    def __init__(self):
        self.mono_left = mono_left
        self.mono_right = mono_right
        self.stereo_cam = stereo_cam
        self.xout_stereo = xout_stereo



    def set_parameters(self):
        self.mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_left.setCamera("left")
        self.mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.mono_right.setCamera("right")

        self.stereo_cam.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT) #Initializing Default Preset
        self.stereo_cam.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3) #Setting median filter with 3x3 Kernel
        self.stereo_cam.setLeftRightCheck(False) #Compute and combine disparities in both L-R and R-L directions, and combine them
        self.stereo_cam.setExtendedDisparity(False) #Disparity range increased from 0-95 to 0-190, combined from full resolution and downscaled images.
        self.stereo_cam.setSubpixel(False) #Compute disparity with sub-pixel interpolation (3 fractional bits by default)
        self.xout_stereo.setStreamName('StereoCameraOut')


    def link_device(self):
        self.mono_left.out.link(self.stereo_cam.left)
        self.mono_right.out.link(self.stereo_cam.right)
        self.stereo_cam.disparity.link(self.xout_stereo.input)


stereo = StereoCamera()
stereo.set_parameters()
stereo.link_device()


class ColorCamera():
    def __init__(self):
        self.color_cam = color_cam
        self.xout_color = xout_color

    def set_parameters(self):
        self.color_cam.setPreviewSize(1280, 720)  #set preview
        self.color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.color_cam.setInterleaved(False) #Choose planar or interleavead representation of data
        self.color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    def link_device(self):
        self.xout_color.setStreamName('ColorCameraOut')
        self.color_cam.preview.link(self.xout_color.input)


color = ColorCamera()
color.set_parameters()
color.link_device()


if __name__ == "__main__":
    with dai.Device(pipeline) as device:
        previewQueue_ColorCamera = device.getOutputQueue('ColorCameraOut', maxSize=4, blocking=False)
        previewQueue_StereoCamera = device.getOutputQueue('StereoCameraOut', maxSize=4, blocking=False)

        def stream_color_camera():
            while True:
                try:
                    if previewQueue_ColorCamera:
                        frame_color = previewQueue_ColorCamera.get()
                        image_color = frame_color.getCvFrame()
                        cv2.imshow('ColorCameraOut', image_color)
                except:
                    print('ColorCamera queue is empty')
                    break;

                if cv2.waitKey(1) == ord('x'):
                    break;

            cv2.destroyAllWindows()

        def stream_stereo_camera():
            while True:
                try:
                    if previewQueue_StereoCamera:
                        frame_stereo = previewQueue_StereoCamera.get()
                        image_stereo = frame_stereo.getFrame()
                        image_stereo = (image_stereo * (255 / stereo_cam.initialConfig.getMaxDisparity())).astype(np.uint8) #normalization and casting to uint8
                        image_stereo = cv2.applyColorMap(image_stereo, cv2.COLORMAP_JET) #setting color map
                        cv2.imshow('StereoCameraOut', image_stereo)
                except:
                    print('StereoCamera queue is empty')
                    break;

                if cv2.waitKey(1) == ord('q'):
                    break;

            cv2.destroyAllWindows()

        stream_stereo_camera()
        stream_color_camera()
