
import cv2
import time

# 定义输入和变量
img1 = cv2.imread('D:\face\wjk.jpg')
img2 = cv2.imread('D:\face\gmlz.jpg')

new_shape = (300, 300)   # 统一缩放为 300*300
cos_thresh = 0.363       # cos阈值，距离越大越接近
L2_thresh = 1.128        # L2阈值，距离越小越接近
img1 = cv2.resize(img1, new_shape)
img2 = cv2.resize(img2, new_shape)

start = time.time()
# 初始化模型：
faceDetector = cv2.FaceDetectorYN.create('D:\face\yunet.onnx', '', new_shape)
faceRecognizer = cv2.FaceRecognizerSF.create('D:\face\face_recognizer_fast.onnx', '')

# 检测、对齐、提取特征：
# detect输出的是一个二维元祖，其中第二维是一个二维数组: n*15,n为人脸数，
# 15为人脸的xywh和5个关键点（右眼瞳孔、左眼、鼻尖、右嘴角、左嘴角）的xy坐标及置信度
faces1 = faceDetector.detect(img1)
aligned_face1 = faceRecognizer.alignCrop(img1, faces1[1][0])    # 对齐后的图片
feature1 = faceRecognizer.feature(aligned_face1)             # 128维特征


faces2 = faceDetector.detect(img2)
aligned_face2 = faceRecognizer.alignCrop(img2, faces2[1][0])
feature2 = faceRecognizer.feature(aligned_face2)

cv2.imwrite('D:\face\aligned31.jpg',aligned_face1)
cv2.imwrite('D:\face\aligned32.jpg',aligned_face2)

# 人脸匹配值打分：
cos_score = faceRecognizer.match(feature1, feature2, 0)
L2_score = faceRecognizer.match(feature1, feature2, 1)

# 输出结果：
print('cos_score: ', cos_score)
print('L2_score: ', L2_score)

if cos_score > cos_thresh:
    print('the same face')
else:
    print('the diffrent face')

if L2_score < L2_thresh:
    print('the same face')
else:
    print('the diffrent face')

end = time.time()
print('all last time:{:.2f} ms'.format(1000*(end - start)))

