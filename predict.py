from ultralytics import YOLO

model = YOLO("my_model.pt")

results = model.predict("testimg/1_video.mp4", show_boxes=True, save=True)
