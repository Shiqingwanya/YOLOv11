from ultralytics import YOLO

model = YOLO("my_model.pt")

results = model.predict("testimg/1.jpg", show_boxes=True, save=True)
