在先前的`RT-DETR`中，博主使用`ONNX`模型文件进行了视频、图像的推理，在本章节，博主打算使用`YOLOv8`模型进行推理，因此，我们除了需要获取`YOLOv8`的`ONNX`模型文件外，还需要进行一些额外的操作，如`NMS`后处理过程，其详细实现过程如下：

# YOLOv8模型导出
在`YOLOv8`的官方项目中，新建`export.py`，写入如下代码即可导出`yolov8n,onnx`文件
```python
from ultralytics import YOLO
model = YOLO("D:\graduate\programs\yolo8/ultralytics-main\yolov8n.pt")
model.export(format="onnx")
```

# Gradio推理UI设计

这里，我们先使用`Gradio`进行推理界面的搭建，其输入与输出均为图像

```python
import YOLODet
import gradio as gr
import cv2
model = 'yolov8n.onnx'
yolo_det = YOLODet.YOLODet(model, conf_thres=0.5, iou_thres=0.3)

def det_img(cv_src):
    yolo_det(cv_src)
    cv_dst = yolo_det.draw_detections(cv_src)
    return cv_dst

if __name__ == '__main__':
    img_input = gr.Image()
    img_output = gr.Image()
    app = gr.Interface(fn=det_img, inputs=img_input, outputs=img_output)
    app.launch()
```

# YOLO目标检测推理
上面已经给出了`YOLOv8`模型推理的函数，那么其具体是如何实现的呢？
上述函数的具体实现分别在`YOLODet.py`与`utils.py`文件中，具体实现过程如下：

## YOLODet实例化

首先是`YOLODet`对象的实例化，其通过`__init__`初始化置信度等参数，并生成`InferenceSession`的实例

```python
yolo_det = YOLODet.YOLODet(model, conf_thres=0.5, iou_thres=0.3)
```
初始化参数代码如下：
```python
def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        # Initialize model
        self.initialize_model(path)
```
生成`InferenceSession`的实例
```python
def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()
```
读取模型的参数信息，`YOLO`模型在训练时设置图像为`640*640`，该信息作为参数保存在了`ONNX`模型文件中，此时通过读取模型文件中的相关参数来为前处理任务设置参数
```python
def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
```
获取ONNX的输出值的名称
```python
def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
```

## 输出结果解析原理

这里的输出值为`output0`，即只有一个值，我们可以通过下面代码来查看这个`onnx`模型的输出结果具体是什么样子的

```python
	import onnx
    # 加载模型
    model = onnx.load('D:\graduate\programs\yolo8/ultralytics-main/runs\detect/train2\weights/best.onnx')
    # 检查模型格式是否完整及正确
    onnx.checker.check_model(model)
    # 获取输出层，包含层名称、维度信息
    output = model.graph.output
    print(output)
```
可以看到其名称即为`output0`，其维度为`3`维，即`（1，7，8400）`
```python
[name: "output0"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 7
      }
      dim {
        dim_value: 8400
      }
    }
  }
}
]
```
这里为何是（`7，8400）`呢，首先解释一下7的原因，前`4`个元素是边界框的坐标`（x, y, w, h）`。
剩下的`3`个元素`（car，truck，bus）`是类别得分。
博主也曾直接使用`YOLOv8`训练好的`pt`模型进行转换，得到的结果为`（1，84，8400）`，其中`84`便是`4`个坐标加上`COCO`数据集中的80个类别
那么，这个8400是如何来的呢，这是由于YOLO的三个不同尺度的检测头的原因：

**(80×80+40×40+20×20)=(6400+1600+400)=8400**

每个网格点产生一个预测结果，即有`8400`个预测结果，同时，关于这一点我们可以通过`后处理`过程来证实


```python
def postprocess(self, input_image, output):
    """
    对模型的输出进行后处理，以提取边界框、置信度分数和类别ID。
    参数:
        input_image (numpy.ndarray): 输入图像。
        output (numpy.ndarray): 模型的输出。
    返回值:
        numpy.ndarray: 带有绘制检测结果的输入图像。
    """
    # 转置并压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组中的行数
    rows = outputs.shape[0]
    # 用于存储检测到的边界框、置信度分数和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    # 计算边界框坐标的缩放因子
    x_factor = self.img_width / self.input_width
    y_factor = self.img_height / self.input_height
    # 遍历输出数组中的每一行
    for i in range(rows):  # 选出大于置信度的检测结果
        # 从当前行中提取类别分数
        classes_scores = outputs[i][4:]
        # 找到类别分数中的最大值
        max_score = np.amax(classes_scores)
        # 如果最大值大于置信度阈值
        if max_score >= self.confidence_thres:
            # 获取具有最高分数的类别ID
            class_id = np.argmax(classes_scores)
            # 从当前行中提取边界框坐标
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
```
注意，该后处理过程通过 `np.amax`确定其所属类别，并将所有大于置信度的结果筛选出，但其数量依旧很多，因此需要进行非极大值抑制`（NMS）`操作。


## YOLO推理代码
推理的代码很简单，只要把图像输入加载好的模型中即可：
得到输出结果`output0`
```python
outputs = self.inference(input_tensor)
```

```python
def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d43280249e7f41d29b1dd67a9b06376f.png)

## 后处理操作之结果解析

随后将输出结果进行后处理操作
```python
self.boxes, self.scores, self.class_ids = self.process_output(outputs)
```
上述的后处理过程实现较为繁杂，因此可以采用如下处理方式：
```python
def process_output(self, output):
        predictions = np.squeeze(output[0]).T#转为维度为（8400，7）
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)#选出预测概率最大的分数
        predictions = predictions[scores > self.conf_threshold, :]#选出大于置信度的预测结果
        scores = scores[scores > self.conf_threshold]#选出大于置信度的分数
        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)#根据每个类别最大的概率得到其对应的类别编号

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)#获取box

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]
```
关于`extract_boxes`函数，其作用为处理 `bounding box`（预测框），主要分为两个过程，一个是将预测框结果恢复到与图像大小相匹配的大小（由于多尺度的关系，其输出的预测框的大小都是归一化后的）。
此外，还要将预测的`（x,y,w,h）`转换为`（x y x y）`形式

```python
def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes
 ```
 
该两部分代码如下：恢复预测框大小

```python
def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
```
将`（x,y,w,h）`转换为`（x y x y）`
```python
def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
```

## 后处理操作值NMS操作
经过上述类别置信度筛选后的结果还可以存在很多，为了让结果更加精确（为了防止一个目标有多个预测框），故进行非极大值抑制，即`NMS`操作，其原理如下：

 1. 对于每个类别，按照预测框的置信度进行排序，将置信度最高的预测框作为基准。
 2. 从剩余的预测框中选择一个与基准框的重叠面积最大的框，如果其重叠面积大于一定的阈值，则将其删除。
 3. 对于剩余的预测框，重复步骤2，直到所有的重叠面积都小于阈值，或者没有被删除的框剩余为止

当然，由于`YOLO`模型的性能较好，我们的置信度在设置为`0.5`时，筛选后的结果便不多了，`8400`个筛选完后仅还有`21`个
```python
indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/945a4bfc10dd496494ea0a9090783abe.png)

代码如下，其过程便是按照每个类别计算保存下的预测框

```python
def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes
```
具体的预测框计算使用如下`nms`函数，这个是针对单个类别进行计算的
```python
def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]#排序

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]#获取分值最高的box
        keep_boxes.append(box_id)#这个box保留

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
		#通过计算IOU来判断是否保留，计算的IOU是与最大框的分数
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
		#去除与最大框的IOU大于阈值的预测框，这些预测框可以认为是预测的同一个目标
        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]
        #提取出保留的预测框的坐标

    return keep_boxes
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/699b2cdf825947e39a3481f375e1e5b0.png)
对于下面的代码

```python
sorted_indices = sorted_indices[keep_indices + 1]
```
为何要加一呢，其实是由于前面计算`IOU`时算的是与最大框的`IOU`，经过`keep_indices = np.where(ious < iou_threshold)[0]`筛选后得到小于阈值的（预测的不是同一个目标）,但此时这个`id`是去除了最大框的，要在`sorted_indices`中选择就需要加一。

计算IOU的代码如下：
```python
def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou
```

## 可视化操作

至此，便可以通过NMS找出最终的预测框了，随后便是在图像上标注出目标了：

```python
cv_dst = yolo_det.draw_detections(cv_src)
```

```python
def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return detections_dog(image, self.boxes, self.scores,
                              self.class_ids, mask_alpha)
```

```python
def detections_dog(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections

    for class_id, box, score in zip(class_ids, boxes, scores):

        color = colors[class_id]

        draw_box(det_img, box, color)
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img
```
最终的检测效果如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce667ac73ba4440cab70718e248c8cec.png)
事实上，这套处理流程不仅可以应对YOLOv8的检测，博主还曾测试过YOLOv9的模型，依旧是可用的。

最终完整代码博主将在完成视频推理设计后公布在`github`，尽情期待。
