from ultralytics import YOLO
import cv2
import requests
import time
from color_camera_stream import ColorCamera
import threading


model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8

# hardcoding the ip adress, because the connection is gonna be on the local network
ip_addr = '192.168.4.1'





class PI_controller:
    def __init__(self):
        self.ip_addr = ip_addr
        self.I = 0
        self._time = 0

    def PI(self, Kp, Ki, x_desired, x_actual):
        now = time.monotonic()
        if self._time == 0:
            dt = 0.05
        else:
            dt = max(1e-3, now - self._time)
        self._time = now

        #compute error
        e = x_desired - x_actual

        #clamp error
        e = max(-255, min(255, e))
        if -3 < e < 3:
            e = 0


        u_ = Ki * self.I + Kp * e
        would_sat = abs(u_) > 255
        push_worse = (u_ * e) > 0

        #accumulate error
        if not (would_sat and push_worse):
            self.I += e * dt

        #calculate PI action
        u = Ki * self.I + Kp * e
        return max(-255, min(255, u))

    def reduce_error_x(self, x_coord, x_center_coord):
        pwm = int(round(self.PI(1, 1, x_center_coord, x_coord)))

        if pwm > 0:
            command = f'{{"T":11,"L":0,"R":{abs(pwm)}}}'
        elif pwm < 0:
            command = f'{{"T":11,"L":{abs(pwm)},"R":0}}'
        else:
            command = f'{{"T":11,"L":0,"R":0}}'

        url = f"http://{self.ip_addr}/js?json={command}"
        response = requests.get(url)
        print(response.text)

    def stop_robot(self):
        command = '{"T":11,"L":0,"R":0}'
        url = f"http://{self.ip_addr}/js?json={command}"
        response = requests.get(url)
        print(response.text)

controller = PI_controller()

camera = ColorCamera()
camera.set_parameters()
camera.link_device()

def run_background():
    threading.Thread(target=camera.start_stream, daemon=True).start()

run_background()


while True:
    try:
        frame = camera.frame
        break
    except ValueError:
        time.sleep(0.01)

print("Camera streaming started. Press 'q' to quit.")

# Main YOLO loop
while True:
    try:
        frame = camera.frame
    except ValueError:
        continue

    results = model.predict(
        source=frame,
        show=False,
        classes=[47],
        max_det=1,
        save=False
    )

    result = results[0] #take the first object for results list
    frame_copy = result.orig_img.copy()


    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            controller.reduce_error_x(x_coord=x_center, x_center_coord=frame_copy.shape[1] // 2)
            cv2.circle(frame_copy, (x_center, y_center), 5, (0, 255, 0), -1)


    cv2.imshow("YOLO Livestream", frame_copy)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
