import os
import json
import numpy as np
import cv2
from pycocotools.mask import decode as coco_rle_decode
from typing import List, Dict
from tqdm import tqdm


# --------------------------
# JSON安全读取（解决非法反斜杠）
# --------------------------
def safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 自动修复非法 \ → \\
    text = text.replace("\\", "\\\\")
    return json.loads(text)


# --------------------------
# 解码 COCO RLE（适配 imageHeight / imageWidth）
# --------------------------
def decode_coco_rle(rle_list: List[Dict], image_shape: tuple) -> List[np.ndarray]:
    if not rle_list:
        raise ValueError("RLE标注列表为空！")

    h, w = image_shape
    instance_masks = []

    for rle in rle_list:
        rle_dict = {
            "counts": rle["counts"],  # 字符串 RLE
            "size": [h, w]            # 必须补充 size
        }
        mask = coco_rle_decode(rle_dict).squeeze()
        if mask.shape != (h, w):
            raise RuntimeError(f"掩码尺寸不一致：{mask.shape} vs 预期 ({h},{w})")
        instance_masks.append(mask.astype(np.uint8))

    return instance_masks


# --------------------------
# mask → polygon（YOLO格式）
# --------------------------
def mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) >= 3:
            polys.append(cnt.reshape(-1, 2))
    return polys


# --------------------------
# 处理单个 JSON → 输出 YOLO 分割 txt
# --------------------------
def process_single_json(json_path: str, save_txt_path: str):
    try:
        data = safe_load_json(json_path)
    except Exception as e:
        print(f"读取 JSON 失败：{json_path} — {str(e)}")
        return

    h = data["imageHeight"]
    w = data["imageWidth"]
    rle_list = data["cocoRLE"]

    try:
        masks = decode_coco_rle(rle_list, (h, w))
    except Exception as e:
        print(f"RLE 解码失败：{json_path} — {str(e)}")
        return

    lines = []

    for rle, mask in zip(rle_list, masks):
        cls = int(rle["label"]) -1  # 从0开始
        polys = mask_to_polygons(mask)

        for poly in polys:
            xy_norm = []
            for x, y in poly:
                xy_norm.append(x / w)
                xy_norm.append(y / h)

            line = str(cls) + " " + " ".join(f"{v:.6f}" for v in xy_norm)
            lines.append(line)

    if len(lines) > 0:
        with open(save_txt_path, "w") as f:
            f.write("\n".join(lines))


# --------------------------
# 批量处理入口
# --------------------------
def batch_convert(json_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for jf in tqdm(json_files, desc="Converting JSON → YOLO TXT"):
        jpath = os.path.join(json_dir, jf)
        tpath = os.path.join(save_dir, jf.replace(".json", ".txt"))

        process_single_json(jpath, tpath)


# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--json_dir", default=r'/new_disk/shy/2025/code/handian/code/datasets/for_exp/train/json', help="存放 json 的文件夹")
    # parser.add_argument("--save_dir", default=r'/new_disk/shy/2025/code/handian/code/datasets/for_exp/train/labels', help="输出 txt 的文件夹")
    parser.add_argument("--json_dir", default=r'/new_disk/shy/2025/code/handian/code/datasets/for_exp/val/json', help="存放 json 的文件夹")
    parser.add_argument("--save_dir", default=r'/new_disk/shy/2025/code/handian/code/datasets/for_exp/val/labels', help="输出 txt 的文件夹")
    args = parser.parse_args()

    batch_convert(args.json_dir, args.save_dir)
    print("全部 JSON 转 YOLO txt 完成！")