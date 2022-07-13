import cv2
import numpy as np
from matplotlib import pyplot as plt

# 图像边界目标检测
class ImageBoundaryDescription:

  # 图像边界目标检测 初始化
  def __init__(self, img):
    # 初始化，读入所需处理的图像
   self.ImgRead(img)

  # 图像的读取
  def ImgRead(self, img):
    self.img = cv2.imread(img)
    return self.img

  # 图像灰度化
  def GrayImg(self):
    # 灰度化
    Gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # 返回灰度图
    return Gray

  # 图像二值化
  def BinaryImg(self, img):
    # 二值化
    # 提高准确性
    '''
    图像二值化,就是将图像的像素点经过阈值(threshold)比较重新设置为0或者255

    cv2.threshold(src, thresh, maxval, type[, dst])
        src: 表示的是图片源
        thresh: 表示的是阈值(起始值)
        maxval: 表示的是最大值
        type: 表示的是这里划分的时候使用的是什么类型的算法**,常用值为0(cv2.THRESH_BINARY)
            cv2.THRESH_BINARY 大于阈值的部分被置为255,小于部分被置为0
            cv2.THRESH_BINARY_INV 大于阈值部分被置为0,小于部分被置为255

    '''
    ret, ImgBinary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 返回二值图
    return ImgBinary

  # 图像轮廓处理
  def ImgContours(self, img):
    '''
    cv2.findContours(img, mode, method)
        mode: 轮廓检索模式
            cv2.RETR_EXTERNAL: 只检索最外面的轮廓；
            cv2.RETR_LIST: 检索所有的轮廓,并将其保存到一条链表当中；
            cv2.RETR_CCOMP: 检索所有的轮廓,并将他们组织为两层:顶层是各部分的外部边界,第二层是空洞的边界;
            cv2.RETR_TREE: 检索所有的轮廓,并重构嵌套轮廓的整个层次;
        method: 轮廓逼近方法
            cv2.CHAIN_APPROX_NONE: 以Freeman链码的方式输出轮廓,所有其他方法输出多边形(顶点的序列).
            cv2.CHAIN_APPROX_SIMPLE: 压缩水平的、垂直的和斜的部分,也就是,函数只保留他们的终点部分.

    '''
    # 连通域分析,寻找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours

  # 图像轮廓绘制
  def DrawContours(self, img, contours, i, color, px):
    '''
    cv2.drawContours(image, contours, contourIdx, color, thickness=None)
        image: 所要绘制的图像
        contours: 所需绘制的轮廓列表
        contourIdx: 所需绘制的轮廓列表的对应轮廓
        color: 绘制轮廓的颜色
        thickness: 绘制轮廓线条的像素宽度,如果是-1,则绘制其中的所有轮廓

    '''
    # (0, 255, 0)
    res = cv2.drawContours(img, contours, i, color, px)

    return res

  # 图像轮廓面积和周长
  def TargetContourSAT(self, TargetContour):
    #面积
    S = cv2.contourArea(TargetContour)

    #周长,True表示闭合的
    T = cv2.arcLength(TargetContour, True) 

    print('目标轮廓的面积：', S)
    print('目标轮廓的周长：', T)
  
  # 图像轮廓最小外接矩形，轮廓形状矩阵
  # 计算边界的直径
  # 获取最小外接矩阵,中心点坐标,宽高,旋转角度
  def TargetContourMinArea(self, TargetContour, img):
    rect = cv2.minAreaRect(TargetContour)
    print('最小外接矩阵 中心点坐标: ', rect[0])
    print('最小外接矩阵 宽: ', rect[1][0])
    print('最小外接矩阵 高: ', rect[1][1])
    print('最小外接矩阵 旋转角度: ', rect[2])

    # 获取矩形四个顶点,浮点型
    box = cv2.boxPoints(rect)

    # 取整
    box = np.int0(box)  # 获得矩形角点

    # area = cv2.contourArea(box)

    width = rect[1][0]
    height = rect[1][1]

    if width > height:
        print('边界的直径: ', width)
    else:
      print('边界的直径: ', height)

    ExternalRectangleImg = img.copy()

    '''

    cv2.polylines(image, [pts], isClosed, color, thickness)
        image: 所要绘制的图像
        pts: 多边形曲行数组.
        npts: 多边形顶点计数器阵列.
        ncontours: 曲行数量.
        isClosed: 指示绘制的折线是否闭合的标志.如果它们是闭合的,则该函数从每条曲线的最后一个顶点到其第一个顶点绘制一条线顶点.
        color: 这是折线的颜色来绘制.对于BGR,我们传递一个元组.
        thickness: 它是折线边的厚度.

    '''
    ExternalRectangle = cv2.polylines(ExternalRectangleImg, [box], True, (0, 255, 0), 2)

    return ExternalRectangle

  # 图像轮廓的直线拟合
  def TargetContourStraightLine(self, TargetContour, img):
    StraightLineImg = img.copy()

    rows, cols = img.shape[:2]

    '''
    output = cv2.fitLine(InputArray points, distType, param, reps, aeps)
    参数：
        InputArray Points: 待拟合的直线的集合，必须是矩阵形式（如numpy.array)
        distType: 距离类型。fitline为距离最小化函数，拟合直线时，要使输入点到拟合直线的距离和最小化。这里的距离的类型有以下几种：
            cv2.DIST_USER : User defined distance
            cv2.DIST_L1: distance = |x1-x2| + |y1-y2|
            cv2.DIST_L2: 欧式距离，此时与最小二乘法相同
            cv2.DIST_C: distance = max(|x1-x2|,|y1-y2|)
            cv2.DIST_L12: L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
        param: 距离参数，跟所选的距离类型有关，值可以设置为0
        reps,aeps: 第5/6个参数用于表示拟合直线所需要的径向和角度精度，通常情况下两个值均被设定为1e-2
    '''
    [vx, vy, x, y] = cv2.fitLine(TargetContour, cv2.DIST_L2, 0, 0.01, 0.01)

    slope = -float(vy) / float(vx)  # 直线斜率

    lefty = int((x * slope) + y)
    righty = int(((x - cols) * slope) + y)

    print('(x1, y1) = ', (0, lefty))
    print('(x2, y2) = ', (cols - 1, righty))

    return cv2.line(StraightLineImg, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

  # 图像展示并保存
  def ShowImg(self, img , imgName, imgSavePath):
    cv2.imshow(imgName, img)
    cv2.waitKey()
    # 'Images/ImgGray.jpeg'
    cv2.imwrite(imgSavePath, img)

def main():
  # 初始化一个 图像边界目标检测 的类
  IBD = ImageBoundaryDescription('xm.jpeg')

  # 读取图像
  img = IBD.ImgRead('xm.jpeg')

  # 图像灰度化
  ImgGray = IBD.GrayImg()

  # 展示并报存灰度图像
  IBD.ShowImg(ImgGray, 'ImgGray', 'Images/ImgGray.jpeg')

  # 图像二值化
  ImgBinary = IBD.BinaryImg(ImgGray)

  # 展示并报存二值图像
  IBD.ShowImg(ImgBinary, 'ImgBinary', 'Images/ImgBinary.jpeg')

  # 轮廓提取
  contours = IBD.ImgContours(ImgBinary)

  # 轮廓数量
  print('轮廓的个数：', len(contours))

  # 定义轮廓颜色,绿色
  color = (0, 255, 0)

  # 保存轮廓
  for i in range(0, len(contours), 1):

    ContourImg = img.copy()
    
    res = IBD.DrawContours(ContourImg, contours, i, color, 2)
    # cv2.imwrite('Images/Contours/Contour' + str(i) + '.jpeg', res)
    IBD.ShowImg(res, 'Contour' + str(i+1), 'Images/Contours/Contour' + str(i+1) + '.jpeg')

  # 记录目标轮廓,即选取所需轮廓
  flag = 1
  TargetContour = contours[flag]

  # 使用红色用于记录所选目标轮廓图像
  TargetContourImg = IBD.DrawContours(ContourImg, contours, flag, (0, 0, 255), 2)

  # 保存目标轮廓图像
  IBD.ShowImg(TargetContourImg, 'TargetContourImg', 'Images/TargetContourImg.jpeg')

  # 计算目标轮廓面积和周长
  IBD.TargetContourSAT(TargetContour)

  # 获取目标轮廓的最小外接矩形
  ExternalRectangle = IBD.TargetContourMinArea(TargetContour, TargetContourImg)

  # 保存目标轮廓的最小外接矩形图像
  IBD.ShowImg(ExternalRectangle, 'ExternalRectangle', 'Images/ExternalRectangleImg.jpeg')

  # 目标轮廓的直线拟合
  StraightLineImg = IBD.TargetContourStraightLine(TargetContour, TargetContourImg)

  # 保存目标轮廓的直线拟合图像
  IBD.ShowImg(StraightLineImg, 'StraightLineImg', 'Images/StraightLineImg.jpeg')

if __name__ == '__main__':
  main()