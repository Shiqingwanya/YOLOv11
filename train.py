from ultralytics import YOLO
# from multiprocessing import Process, freeze_support, set_start_method
#
#
# if __name__ == '__main__':
#     freeze_support()
#     set_start_method('spawn')
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
results = model.train(data="dataset.yaml", epochs=100, imgsz=640, device="0", workers=0, save=True)

val = model.val()
print(val.box.map)

model.save("my_model.pt")

