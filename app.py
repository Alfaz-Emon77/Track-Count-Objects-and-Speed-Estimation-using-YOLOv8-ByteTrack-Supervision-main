from ultralytics import YOLO
import cv2
from time import time
import numpy as np
from Pyresearch import BirdsEyeView
import colorsys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

cap = cv2.VideoCapture("1860079-uhd_2560_1440_25fps.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
source_fps = int(cap.get(cv2.CAP_PROP_FPS))
print("frame_width:", frame_width)
print("frame_height:", frame_height)
print("Source FPS:", source_fps)

if not cap.isOpened():
    print("Error Opening Video File.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video/output_video.mp4", fourcc, source_fps, (frame_width, frame_height))

def color(tracking_id):
    hue = (tracking_id * 137.5) % 360
    red, green, blue = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)

model = YOLO("yolov8n.pt")

road_polygon_points = [(0, 0), (0, 789), (3300, int(789)), (3300, 0), (0, 0)]

target_width = 50
target_height = 250
source = np.array([[1252, 789], [2289, 789], [5039, 2159], [-550, 2159]])
target = np.array([[0, 0], [target_width-1, 0], [target_width-1, target_height-1], [0, target_height-1]])
transformation = BirdsEyeView(source, target, frame_width, frame_height, target_width, target_height)

prev_y_dict = {}
speeding_vehicles = {}
trails = {}

output_dir = "speeding_images"
os.makedirs(output_dir, exist_ok=True)

ptime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_for_ticket = frame.copy()

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(road_polygon_points, dtype=np.int32)], (255, 255, 255))
    original_region = cv2.bitwise_and(frame, mask)
    mask_inv = cv2.bitwise_not(mask)
    frame = cv2.bitwise_and(frame, mask_inv)
    frame = transformation.draw_canvas_boundary(frame)
    frame = transformation.draw_road_lines(frame)  # Removed background fill to avoid black region

    results = model.track(
        frame,
        conf=0.3,
        imgsz=(1280, 736),
        persist=True,
        classes=[2, 7],
        tracker="botsort.yaml",
        verbose=False
    )

    for result in results:
        for r in result.boxes.data.tolist():
            if len(r) < 7:
                continue
            x1, y1, x2, y2, track_id, conf, class_id = map(int, r[:7])
            color_id = color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 2)
            label = f"{'car' if class_id == 2 else 'truck'} {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_id, 2)

            if track_id not in trails:
                trails[track_id] = []
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            trails[track_id].append(center)
            if len(trails[track_id]) > 10:
                trails[track_id].pop(0)
            for i in range(1, len(trails[track_id])):
                cv2.line(frame, trails[track_id][i-1], trails[track_id][i], color_id, 2)

            bottom_center = (int((x1 + x2) / 2), y2)
            transformed_point = transformation.transform_points(np.array([bottom_center]))[0]

            if track_id not in prev_y_dict:
                prev_y_dict[track_id] = []
            prev_y_dict[track_id].append(transformed_point[1])

            if len(prev_y_dict[track_id]) > (source_fps // 2):
                speed = transformation.speed_calculation(prev_y_dict[track_id], source_fps)
                prev_y_dict[track_id].pop(0)

                label_width = int((x2 - x1) * 0.45)
                bbox_center = int((x1 + x2) / 2)
                cv2.rectangle(frame, (bbox_center-label_width, y2+10), (bbox_center+label_width, y2+40), color_id, -1)
                cv2.putText(frame, f"{speed} Km/h", (bbox_center-label_width, y2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                if (class_id == 2 and speed > 120) or (class_id == 7 and speed > 100):
                    if track_id not in speeding_vehicles:
                        speeding_vehicles[track_id] = {"speed": speed, "images": []}
                    image_filename = f"{output_dir}/speeding_{track_id}_{speed}kmh.jpg"
                    cv2.imwrite(image_filename, frame_for_ticket[y1:y2, x1:x2])
                    speeding_vehicles[track_id]["images"].append((image_filename, speed))

    ctime = time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f"FPS: {fps}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

for track_id, data in speeding_vehicles.items():
    pdf_path = f"{output_dir}/ticket_{track_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "স্পীডিং ভেহিকল টিকেট")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"ভেহিকল আইডি: {track_id}")
    c.drawString(100, 700, f"গতি: {data['speed']} km/h")

    y_position = 650
    for img_path, speed in data["images"]:
        if os.path.exists(img_path):
            c.drawImage(img_path, 100, y_position-150, width=400, height=200)
            c.drawString(100, y_position-170, f"গতি: {speed} km/h")
            y_position -= 250

    c.save()
