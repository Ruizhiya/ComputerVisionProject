**Python + OpenCV 基于图像边界的目标检测**

---
[TOC]
---
# **前言**

# **项目总览**

## 所需处理的图片
![cat](xm.jpeg)

## 图像灰度化
![ImgGray](Images/ImgGray.jpeg)

## 图像二值化
![ImgBinary](Images/ImgBinary.jpeg)

## 轮廓提取以及相关处理

### 部分轮廓展示(此处展示提取到的前三个轮廓)

![Contour1](Images/Contours/Contour1.jpeg)
![Contour2](Images/Contours/Contour2.jpeg)
![Contour3](Images/Contours/Contour3.jpeg)

### 标记将要处理的轮廓
![TargetContourImg](Images/TargetContourImg.jpeg)

### 获取目标轮廓的最小外接矩形
![TargetContourImg](Images/ExternalRectangleImg.jpeg)

### 获取目标轮廓的拟合直线
![TargetContourImg](Images/StraightLineImg.jpeg)
