from ultralytics.models import RTDETR
from  ultralytics.utils.plotting import Annotator
import os
import glob
import cv2


def get_next_experiment_folder(base_path):
    existing_experiments = glob.glob(os.path.join(base_path, 'exp*'))
    if not existing_experiments:
        return os.path.join(base_path, 'exp1')
    else:
        experiment_numbers = [int(exp.split('exp')[-1]) for exp in existing_experiments]
        next_experiment_number = max(experiment_numbers) + 1
        return os.path.join(base_path, f'exp{next_experiment_number}')


if __name__ == '__main__':
    model = RTDETR(model=r'E:\Space-python\Space-5\RT-DETR\runs\detect\train\weights\best.pt')

    # 定义输出目录
    base_output_dir = r'E:\Space-python\Space-5\RT-DETR\outputs'
    output_dir = get_next_experiment_folder(base_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 进行预测
    results = model.predict(source=r'E:\Space-python\Space-5\RT-DETR\img', batch=16, device='0', imgsz=512,
                            workers=8)

    # 保存预测结果
    for i, result in enumerate(results):
        # 使用默认绘图方法绘制带有标签的图片
        # annotated_img = result.plot(font_size=3)  # 绘制带有标签的图片
        # cv2.imwrite(os.path.join(output_dir, f'predicted_image_{i}.jpg'), annotated_img)  # 使用OpenCV保存图片
        if result.boxes.data.shape[0] > 0:  # 确保有检测框
            annotator = Annotator(result.orig_img, line_width=1,font_size=2)  # 设置字体大小
            for box in result.boxes.data:  # 假设 boxes 是你的预测框
                x1, y1, x2, y2, conf, class_id = box[:6]  # 获取框的坐标和置信度
                label = f"ship: {conf:.2f}"  # 格式化标签，包括置信度
                annotator.box_label(box=[x1, y1, x2, y2], label=label, color=(0, 0, 255))  # 红色的BGR格式

            annotated_img = annotator.im  # 获取标注后的图片
            cv2.imwrite(os.path.join(output_dir, f'predicted_image_{i}.jpg'), annotated_img)  # 使用OpenCV保存图片
        else:
            print(f"No detections for image {i}, skipping...")