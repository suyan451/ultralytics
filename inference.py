from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("/new_disk/shy/2025/code/handian/code/runs/segment/train3/weights/best.pt")  # load an official model
# model = YOLO("/new_disk/shy/2025/code/handian/code/runs/segment/train2/weights/best.pt")  # load a custom model
# model = YOLO("yolo11n-seg.pt")  # load a custom model

# Predict with the model
# results = model("/new_disk/shy/2025/code/handian/code/bus.jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/20_0.0.27.jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/24007700_20250717162654063.jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/24007779_20250717162613043.jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/20250605235339147.jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/RS0O12M10GY015530A0M18501C_No1_Img12_ML_1#_RS0O12M10GY015530A0M18501C_20250607170552805023_Src.jpg")  # predict on an image
results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/Ant (19).jpg")  # predict on an image
# results = model("/new_disk/shy/2025/code/handian/code/datasets/for_exp_collect/images/val/OK (4).jpg")  # predict on an image

# Access the results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()
    # result.save(filename="Ant (19)_fit.png", boxes=False)  # save to disk
    result.save(filename="Ant (19)_fit1.png", boxes=True)  # save to disk
    # result.save(filename="20_0.0.27_fit.png")  # save to disk