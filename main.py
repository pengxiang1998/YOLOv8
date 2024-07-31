import YOLODet
import gradio as gr
import cv2
#from vis import demo_gradio
model = 'yolov8n.onnx'

base_conf,base_iou=0.5,0.3

def det_img(cv_src,conf_thres, iou_thres):
    yolo_det = YOLODet.YOLODet(model, conf_thres=conf_thres, iou_thres= iou_thres)
    yolo_det(cv_src)
    cv_dst = yolo_det.draw_detections(cv_src)
    return cv_dst

def detectio_video(input_path):
    yolo_det = YOLODet.YOLODet(model, conf_thres=0.5, iou_thres=0.3)
    output_path="result.mp4"
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(5))
    t = int(1000 / fps)
    videoWriter = None
    while True:
        _, img = cap.read()
        if img is None:
            break
        yolo_det(img)
        cv_dst = yolo_det.draw_detections(img)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(output_path, fourcc, fps, (cv_dst.shape[1], cv_dst.shape[0]))
        videoWriter.write(cv_dst)
    cap.release()
    videoWriter.release()
    return output_path
def moviepy_video(input_path):
    from moviepy.editor import ImageSequenceClip
    yolo_det = YOLODet.YOLODet(model, conf_thres=0.5, iou_thres=0.3)
    output_path="result.mp4"
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(5))
    t = int(1000 / fps)
    images=[]
    while True:
        _, img = cap.read()
        if img is None:
            break
        yolo_det(img)
        cv_dst = yolo_det.draw_detections(img)
        images.append(cv_dst)
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_path, codec='libx264')
    clip.close()
    cap.release()
    return output_path


if __name__ == '__main__':

    img_input = gr.Image()
    img_output = gr.Image()
    video_input=gr.Video(sources="upload")
    app1 = gr.Interface(fn=det_img, inputs=[img_input,
                                            gr.Slider(maximum=1,minimum=0,value=base_conf),
                                            gr.Slider(maximum=1,minimum=0,value=base_iou)], outputs=img_output)
    app2 = gr.Interface(fn=moviepy_video, inputs=video_input, outputs="video")
    demo = gr.TabbedInterface(
        [app1, app2],
        tab_names=["图像目标检测", "视频目标检测"],
        title="目标检测"
    )

    demo.launch()
