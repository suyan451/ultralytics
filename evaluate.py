from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("/new_disk/shy/2025/code/handian/code/runs/segment/train2/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list containing mAP50-95(B) for each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list containing mAP50-95(M) for each category