import depthai as dai
import cv2
import numpy as np
import threading
import time

# Shared latest depth frame (mm), aligned to RGB
_latest_depth_mm = None
_lock = threading.Lock()

def get_latest_depth_mm():
    with _lock:
        return None if _latest_depth_mm is None else _latest_depth_mm.copy()

class StereoCamera:
    """
    Provides a depth stream (millimeters) aligned to RGB resolution.
    You can display it optionally; object_detection.py just reads get_latest_depth_mm().
    """
    def __init__(self, rgb_size=(640, 400), median_kernel=dai.MedianFilter.KERNEL_3x3, show=False):
        self.rgb_w, self.rgb_h = rgb_size
        self.median_kernel = median_kernel
        self.show = show
        self._running = False

    def _build_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB camera (for alignment target only)
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.rgb_w, self.rgb_h)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        # Stereo
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera("left")
        mono_right.setCamera("right")

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setSubpixel(True)
        stereo.initialConfig.setMedianFilter(self.median_kernel)

        # Align the depth to the RGB camera
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Depth output
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # We don’t export RGB here (object_detection gets RGB from ColorCamera class)
        return pipeline

    def start_stream(self):
        """
        Runs in a background thread. Fills _latest_depth_mm with millimeter depth aligned to RGB.
        """
        if self._running:
            return
        self._running = True

        pipeline = self._build_pipeline()
        try:
            with dai.Device(pipeline) as device:
                q_depth = device.getOutputQueue('depth', maxSize=4, blocking=False)

                while self._running:
                    depth_pkt = q_depth.tryGet()
                    if depth_pkt is not None:
                        frame = depth_pkt.getFrame()  # uint16, millimeters, aligned to RGB
                        # Optional small normalization for display
                        if self.show:
                            disp = frame.astype(np.float32)
                            disp[disp <= 0] = np.nan
                            # simple visualization: clip to 4 m
                            vmax = 4000.0
                            disp = np.clip(disp, 0, vmax) / vmax
                            vis = (disp * 255.0).astype(np.uint8)
                            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                            cv2.imshow('Stereo Depth (aligned)', vis)
                            if cv2.waitKey(1) == ord('q'):
                                break
                        with _lock:
                            global _latest_depth_mm
                            _latest_depth_mm = frame.copy()
                    else:
                        time.sleep(0.005)
        finally:
            self._running = False
            if self.show:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass




