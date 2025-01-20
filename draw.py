import os
import cv2
import numpy as np

# 路径配置
label_dir = r"F:\github_repo\ICAFusion-main\runs\test\exp\labels"  # 标签文件夹
# label_dir = r"F:\github_repo\ICAFusion-main\VEDAI\labels\test"  # 标签文件夹
image_dir = r"F:\github_repo\ICAFusion-main\VEDAI\visible\test"  # 图像文件夹
output_dir = r"F:\github_repo\ICAFusion-main\runs\test\exp\image1"  # 输出文件夹

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 类别名称映射
class_names = {
    0: "car",
    1: "truck",
    2: "pickup",
    3: "tractor",
    4: "camper",
    5: "ship",
    6: "van",
    7: "plane"
}

# 类别颜色映射（每个类别一个独特的颜色）
class_colors = {
    0: (255, 0, 0),  # 红色
    1: (0, 255, 0),  # 绿色
    2: (0, 0, 255),  # 蓝色
    3: (255, 255, 0),  # 黄色
    4: (255, 0, 255),  # 粉色
    5: (0, 255, 255),  # 青色
    6: (128, 0, 128),  # 紫色
    7: (255, 165, 0)  # 橙色
}


# NMS函数
def non_max_suppression(boxes, scores, iou_threshold=0.45, method='standard', sigma=0.5, score_threshold=0.25):
    """
    非极大值抑制（NMS）
    :param boxes: 边界框列表，格式为 [x1, y1, x2, y2]
    :param scores: 每个框的置信度
    :param iou_threshold: IoU阈值，大于该值的框会被抑制
    :param method: NMS方法，支持 'standard'（标准NMS）和 'soft'（Soft-NMS）
    :param sigma: Soft-NMS的参数，用于控制分数衰减
    :param score_threshold: 置信度阈值，低于该值的框会被过滤
    :return: 保留的框的索引
    """
    if len(boxes) == 0:
        return []

    # 将框坐标和置信度转换为 NumPy 数组
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # 过滤掉置信度低于阈值的框
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return []

    # 获取框的左上角和右下角坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按置信度排序
    order = scores.argsort()[::-1]

    keep_indices = []  # 保留的框的索引
    while order.size > 0:
        i = order[0]  # 当前置信度最高的框
        keep_indices.append(i)

        # 计算当前框与其他框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        if method == 'standard':
            # 标准NMS：保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
        elif method == 'soft':
            # Soft-NMS：根据IoU衰减分数
            scores[order[1:]] *= np.exp(-iou**2 / sigma)
            inds = np.where(scores[order[1:]] >= score_threshold)[0]
        else:
            raise ValueError(f"Unknown NMS method: {method}")

        order = order[inds + 1]

    return keep_indices


# 遍历标签文件夹
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        # 获取标签文件的完整路径
        label_path = os.path.join(label_dir, label_file)

        # 获取对应的图像文件名（假设图像文件名与标签文件名相同，只是扩展名不同）
        image_name = os.path.splitext(label_file)[0] + ".png"  # 假设图像是PNG格式
        image_path = os.path.join(image_dir, image_name)

        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件 {image_name} 不存在，跳过")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件 {image_name}，跳过")
            continue

        # 获取图像的宽度和高度
        img_height, img_width = image.shape[:2]

        # 读取标签文件
        with open(label_path, "r") as f:
            lines = f.readlines()

        # 解析标签
        boxes = []  # 边界框列表
        scores = []  # 置信度列表
        class_ids = []  # 类别ID列表
        for line in lines:
            # 动态判断分隔符（逗号或空格）
            if "," in line:
                values = line.strip().split(",")  # 使用逗号分隔
            else:
                values = line.strip().split()  # 使用空格分隔

            if len(values) < 5:
                print(f"标签格式错误：{line.strip()}，跳过")
                continue

            # 提取数值
            class_id = int(float(values[0]))  # 类别ID
            x_center = float(values[1])  # x_center
            y_center = float(values[2])  # y_center
            width = float(values[3])  # width
            height = float(values[4])  # height
            confidence = float(values[5]) if len(values) > 5 else 1.0  # 置信度（如果没有则默认为1.0）

            # 如果坐标是归一化的，转换为实际像素值
            if x_center < 1 and y_center < 1 and width < 1 and height < 1:
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

            # 计算边界框的左上角和右下角坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # 添加到列表
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)

        # 应用NMS
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.45, method='standard')
        # keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.45, method='soft', sigma=0.5)

        # 绘制保留的框
        for i in keep_indices:
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            confidence = scores[i]

            # 获取类别名称和颜色
            class_name = class_names.get(class_id, f"Class {class_id}")
            color = class_colors.get(class_id, (0, 255, 0))  # 默认绿色

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 在框上方显示类别名称和置信度
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存结果图像
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image)
        print(f"已保存结果图像到 {output_path}")

print("处理完成！")