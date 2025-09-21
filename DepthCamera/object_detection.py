from ultralytics import YOLO
import cv2
import requests
import time
import threading
import numpy as np

from color_camera_stream import ColorCamera
from stereo_camera_stream import StereoCamera, get_latest_depth_mm


model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8

# hardcoding the ip adress, because the connection is gonna be on the local network
ip_addr = '192.168.4.1'

# ------------------ control tuning ------------------
TARGET_M = 0.50                  # target stand-off distance (meters)
DEADBAND_PIX = 3                 # no yaw correction if |x error| < 3 px
DEADBAND_M   = 0.03              # no forward correction if |range error| < 3 cm
MAX_PWM = 255
PWM_MIN = 70                     # minimum effective PWM to overcome static friction

Kp_yaw, Ki_yaw = 0.45, 0.06      # yaw PI
Kp_fwd, Ki_fwd = 0.60, 0.10      # forward PI

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def mix_lr(throttle, yaw):
    """
    throttle >=0 forward effort (0..255)
    yaw: right positive, left negative (-255..255)
    Returns (L, R) each in 0..255
    """
    L = clamp(int(round(throttle - yaw)), -MAX_PWM, MAX_PWM)
    R = clamp(int(round(throttle + yaw)), -MAX_PWM, MAX_PWM)
    # Controller expects non-negative L/R
    return max(0, L), max(0, R)

def apply_deadzone_pwm(pwm):
    if pwm == 0:
        return 0
    sign = 1 if pwm > 0 else -1
    return sign * max(PWM_MIN, abs(pwm))


class PI_controller:
    def __init__(self):
        self.ip_addr = ip_addr
        self.I_yaw = 0.0
        self.I_fwd = 0.0
        self._time = 0.0

    def _dt(self):
        now = time.monotonic()
        if self._time == 0:
            dt = 0.05
        else:
            dt = max(1e-3, now - self._time)
        self._time = now
        return dt

    @staticmethod
    def _anti_windup_integrate(I, e, Ki, u, umax, dt):
        """
        Back-calculation style: stop integrating if the *unsaturated* control would
        push further into saturation in the same direction as the error.
        """
        would_sat = abs(u) > umax
        push_worse = (u * e) > 0
        if not (would_sat and push_worse):
            I += e * dt
        return I

    def control(self, x_desired, x_actual, z_meas_m):
        """
        Returns motor pair (L,R) 0..255 based on horizontal pixel error and depth error.
        """
        dt = self._dt()

        # --------- YAW (pixel) ----------
        e_yaw = (x_desired - x_actual)
        if -DEADBAND_PIX < e_yaw < DEADBAND_PIX:
            e_yaw = 0.0

        u_yaw_unsat = Kp_yaw * e_yaw + Ki_yaw * self.I_yaw
        self.I_yaw = self._anti_windup_integrate(self.I_yaw, e_yaw, Ki_yaw, u_yaw_unsat, MAX_PWM, dt)
        u_yaw = clamp(Kp_yaw * e_yaw + Ki_yaw * self.I_yaw, -MAX_PWM, MAX_PWM)

        # --------- FORWARD (meters) ----------
        if z_meas_m is None or np.isnan(z_meas_m) or z_meas_m <= 0:
            # no valid range → only do yaw, no forward
            u_fwd = 0.0
        else:
            e_fwd = (z_meas_m - TARGET_M)  # positive if too far → drive forward (+)
            if abs(e_fwd) < DEADBAND_M:
                e_fwd = 0.0

            u_fwd_unsat = Kp_fwd * e_fwd + Ki_fwd * self.I_fwd
            self.I_fwd = self._anti_windup_integrate(self.I_fwd, e_fwd, Ki_fwd, u_fwd_unsat, MAX_PWM, dt)
            u_fwd = clamp(Kp_fwd * e_fwd + Ki_fwd * self.I_fwd, 0, MAX_PWM)  # forward only

        # Deadzone compensation
        if u_fwd > 0:
            u_fwd = apply_deadzone_pwm(int(round(u_fwd)))
        u_yaw = int(round(u_yaw))
        if u_yaw != 0:
            u_yaw = apply_deadzone_pwm(u_yaw)

        L, R = mix_lr(u_fwd, u_yaw)
        return L, R

    def send(self, L, R):
        command = f'{{"T":11,"L":{int(L)},"R":{int(R)}}}'
        try:
            url = f"http://{self.ip_addr}/js?json={command}"
            requests.get(url, timeout=0.15)
        except Exception as ex:
            print("HTTP error:", ex)

    def stop_robot(self):
        try:
            requests.get(f"http://{self.ip_addr}/js?json={{\"T\":11,\"L\":0,\"R\":0}}", timeout=0.15)
        except Exception as ex:
            print("HTTP error:", ex)


controller = PI_controller()

# ---------- start cameras ----------
camera = ColorCamera()
camera.set_parameters()
camera.link_device()

stereo = StereoCamera()  # depth aligned to RGB, exported as global latest frame

def run_background():
    threading.Thread(target=camera.start_stream, daemon=True).start()
    threading.Thread(target=stereo.start_stream, daemon=True).start()

run_background()

# wait for first RGB frame
while True:
    try:
        frame = camera.frame
        break
    except ValueError:
        time.sleep(0.01)

print("RGB + Depth streaming started. Press 'q' to quit.")

PATCH_R = 5  # 5x5 depth patch half-size

# Main YOLO loop
while True:
    try:
        frame = camera.frame
    except ValueError:
        continue

    results = model.predict(
        source=frame,
        show=False,
        classes=[47],  # change to your class IDs
        max_det=1,
        save=False,
        verbose=False
    )

    result = results[0]
    frame_copy = result.orig_img.copy()
    h, w = frame_copy.shape[:2]
    cx_desired = w // 2

    z_m = None  # default if no detection

    if result.boxes is not None and len(result.boxes) > 0:
        # Take top-1 box
        box = result.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        cv2.circle(frame_copy, (x_center, y_center), 5, (0, 255, 0), -1)

        # ------- depth at detection center (median of small patch) -------
        depth_mm = get_latest_depth_mm()
        if depth_mm is not None:
            y0 = max(0, y_center - PATCH_R)
            y1p = min(depth_mm.shape[0], y_center + PATCH_R + 1)
            x0 = max(0, x_center - PATCH_R)
            x1p = min(depth_mm.shape[1], x_center + PATCH_R + 1)
            patch = depth_mm[y0:y1p, x0:x1p]
            # Filter out zeros (invalid)
            patch_valid = patch[(patch > 0) & (patch < 65535)]
            if patch_valid.size > 0:
                z_m = float(np.median(patch_valid)) / 1000.0  # mm → meters
                cv2.putText(frame_copy, f"{z_m:.2f} m", (x_center + 8, y_center - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ------- compute and send control -------
        L, R = controller.control(cx_desired, x_center, z_m)
        controller.send(L, R)
    else:
        # No detection → stop (or keep last yaw only if you prefer)
        controller.stop_robot()

    cv2.imshow("YOLO Livestream", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        controller.stop_robot()
        break

cv2.destroyAllWindows()












# for results in model.predict(source=frame_generator(), stream=True, show=True, classes=[47], save=False, max_det=1):
#     # creating naked image(just taking image from camera)
#     frame = results.orig_img.copy()
#     print(model.names)
#     # checking whether or not there are some bb, if not, just show regular image
#     if results.boxes is None or len(results.boxes) == 0:
#         cv2.imshow("YOLO livestream", frame)
#         #stop_robot()
#
#     else:
#         # if some bb was detecter using formulas we are calculating the center of the bb
#
#         original_cords = results.boxes.data
#         # calculating the center of the bounding box.
#         x_center = (original_cords[0][0] + (original_cords[0][2] - original_cords[0][0]) / 2)
#         y_center = (original_cords[0][1] + (original_cords[0][3] - original_cords[0][1]) / 2)
#         center_of_bb = [int(x_center.item()), int(y_center.item())]
#         print(center_of_bb)
#         reduce_error_x(320, center_of_bb[0])
#         # drawing dot over the "frame" image
#         cv2.circle(frame, center=(center_of_bb[0], center_of_bb[1]), radius=1, color=(0, 255, 0), thickness=5)
#     # showing what
#     cv2.imshow("YOLO livestream", frame)
#     # handaling exceptions
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()