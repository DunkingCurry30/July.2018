#OpenCV
import cv2

#####
# 1 #读入图像（读入，显示，保存）
#####


'''img = cv2.imread("soccer.jpg")#不支持中文路径
img1 = cv2.imread("soccer.jpg",cv2.IMREAD_COLOR)#读入彩色图像，默认参数
img2 = cv2.imread("soccer.jpg",cv2.IMREAD_GRAYSCALE)#以灰度模式读入
cv2.namedWindow('img',cv2.WINDOW_NORMAL)#设置可变窗口
cv2.imshow('img',img)#显示图像第一个参数为窗口名，第二个为对象
#cv2.imshow('img1',img1)
#cv2.imshow('img2',img2)
cv2.waitKey(0)#绑定键盘,0为无限等待键盘输入
#cv2.imwrite("c.jpg",img)#保存,不支持中文
'''

#实例1
'''img = cv2.imread("soccer.jpg",cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k ==ord('s'):
    cv2.imwrite("c.jpg",img)
    cv2.destroyAllWindows()
'''
#使用matplotlib显示图像
'''from matplotlib import pyplot as plt
img = cv2.imread("soccer.jpg")#加载彩色图片会出现问题(OpenCV为BGR，而Mat为RGB）
#还原彩色的解决方法
b,g,r = cv2.split(img)#得到三个通道
img2 = cv2.merge([r,g,b])#置换顺序
plt.imshow(img2,cmap = 'gray',interpolation = 'bicubic')
plt.xticks([]),plt.yticks([])
plt.show()'''


#####
# 2 #视频
#####


#使用摄像头捕获视频
'''cap = cv2.VideoCapture(0)#创建一个VideoCapture对象,参数0使用默认摄像头,也可以是
cap1 = cv2.VideoCapture("skylight.mov")#一个视屏文件

while(True):
    ret,frame = cap1.read()#读取帧无误返回True
    #print(cap.isOpened())#使用isOpened()测试是否开启摄像头
    #cap.open(0)#手动开启摄像头
    print(cap.get(3))#帧宽，cap.set(3,320)来设置
    print(cap.get(4))#帧高
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度

    cv2.imshow('frame',gray)
    if cv2.waitKey(0) == ord('q'):
        cv2.imwrite("camera.jpg",gray)
        break
cap.release()#停止逐帧捕获
cv2.destroyAllWindows()
'''

#保存视频
'''cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')#'XVID'为编码格式
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))#每秒20帧保存到output.avi,

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)#0参数绕x轴旋转每一帧，1绕y轴，-1同时翻转
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

ret,frame = cap.read()'''


#####
# 3 # 绘图函数
#####


#color:形状颜色，传入一个类似（255,0,0)BGR的元组
#thickness:线条粗细，默认1,-1代表闭合图形填充
#linetype：线条类型，cv2.LINE_AA为抗锯齿
#import numpy as np

#基本图形
'''img = np.zeros((512,512,3),np.uint8)
cv2.line(img,(0,0),(511,511),(255,0,0),1)#绘图函数返回值都为None
cv2.circle(img,(447,63),63,(123,123,123),-1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),2,cv2.LINE_AA)#放上文字
cv2.imshow('draw',img)'''
'''
#根据选择模式，拖动鼠标画矩形或圆圈
import numpy as np
drawing = False
mode = True#True为矩形
ix,iy = -1,-1

#创建回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:#左键按下
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                r = int(np.sqrt((x-ix)**2+(y-iy)**2)/2)
                cv2.circle(img,(x,y),r,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = Falseq

img =np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == ord('q'):
        break

'''
#滑动块做调色板
'''
#回调函数
import numpy as np
def nothing(x):
    pass

img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')


cv2.createTrackbar('R','image',0,255,nothing)#第一个参数为滑块名称
cv2.createTrackbar('G','image',0,255,nothing)#第二个参数为窗口名称
cv2.createTrackbar('B','image',0,255,nothing)#第三，四个参数默认值，最大值

switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)#滑块充当按钮
while(1):
    
    
    
    cv2.imshow('image',img)
    k = cv2.waitKey(1)
    if k == 27:
        break

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s==0:
        img[:]=255
    else:
        img[:]=[b,g,r]

cv2.destroyAllWindows()
'''
'''
#综合练习

drawing = False
def nothing(x):
    pass
def draw(event,x,y,flags,param):
    global drawing,color
    ix,iy = -1,-1
    if event == cv2.EVENT_LBUTTONDOWN:#左键按下
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.circle(img,(x,y),3,color,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.full((512,512,3),255,np.uint8)#uint8为无符号整数(0~255)
cv2.namedWindow('image')               #full()以指定数填充数组

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)



while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(1) == 27:
        break
    
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')

    color = [b,g,r]
    cv2.setMouseCallback('image',draw)

cv2.destroyAllWindows()
'''

#####
# 4 # 图像基础
#####
'''
import numpy as np
img = cv2.imread("soccer.jpg")#返回一个三维数组，可使用下标访问
'''
#获取图像属性
'''
img.itemset((10,10,2),100)#修改像素
print(img.shape,img.size) #返回行数，列数，通道数的元组；返回像素数目
print(img.dtype)##返回图像数据类型
'''

#图像ROI
'''
area = img[100:160,220:290]#对图像的一部分区域进行拷贝
img[330:390,440:510] = area

cv2.imshow('ROI',img)
'''

#拆分/合并图像通道
'''
b,g,r = cv2.split(img)#分离通道,较慢
img[:,:,2] = 0#Numpy索引，红色通道全为0，较快,推荐此方法
img1 = cv2.merge([b,g,r])#合并通道

cv2.imshow('img',img)
cv2.imshow('img1',img1)
'''

#为图像扩边（填充）
'''from matplotlib import pyplot as plt

RED=[255,0,0]
#切换成RGB
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

#六种填充效果演示
replicate = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE)#四个10代表边界值
reflect = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)#top,bottom,left,right
reflect101 = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT_101)#最后一个参数
wrap = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_WRAP)#为borderType
constant = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')#subplot(nrows,ncols,index)
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')#三个参数分别为行数，列数，位置
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')#意为2x3表格中的第3个位置
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')#subplot(2,3,4)同（234）
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()
'''

#图像上的算术运算
#图像加法
'''
x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x,y))#OpenCV的加法为饱和操作，结果为255
print(x+y)#Numpy加法为取模，结果为 260%256 = 4.

#图像混合
img2 = cv2.imread("summer.jpg")
#两张图片大小必须一样
img2 = img2[:600,:811]

mix = cv2.addWeighted(img,0.4,img2,0.6,0)#0.7,0.3分别为权重，具体看help(cv2.addWeighter)
cv2.imshow('mix',mix)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#按位运算
#AND,OR,NOT,XOR
#用于选择非矩形ROI
'''img2 = cv2.imread('OpenCV_logo.jpg')

rows,cols,channels = img2.shape
roi = img[:rows,:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)#转换为灰度图
#cv2.imshow('res_gray',img2gray)

#175为阈值，255为最大值，最后为阈值类型
#cv2.THRESH_BINARY表示大于阈值取最大值，小于阈值取0
ret,mask = cv2.threshold(img2gray,168,255,cv2.THRESH_BINARY)
#bitwise_not按位非运算，取反
mask_inv = cv2.bitwise_not(mask)

#这里的mask参数类似ps中的遮罩，控制处理区域
img_bg = cv2.bitwise_and(roi,roi,mask=mask)
#cv2.imshow('res2',img_bg)
img2_bg = cv2.bitwise_and(img2,img2,mask = mask_inv)
#cv2.imshow('res3',img2_bg)

#图像叠加得到原图除白色以外的区域
dst = cv2.add(img_bg,img2_bg)
img[:rows,:cols] = dst

cv2.imshow('res',img)
cv2.imwrite('logoMask.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#程序性能检测
#效率优化：避免循环、算法尽量使用向量、使用视图代替复制数组
#一般情况下OpenCv函数比Numpy快，除了对视图进行操作时
'''import time

img1 = cv2.imread('soccer.jpg')
#查看默认优化是否开启
cv2.useOptimized()
#设置默认优化（没察觉到效果）
cv2.setUseOptimized(False)

e1 = cv2.getTickCount()#记录当前时间
for i in range(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()

t = (e2-e1)/cv2.getTickFrequency()#后一个参数为时钟周期
print(t)
'''

#####
# 5 # 图像处理
#####

#颜色空间转换
#主要两种BGR<->Gray和BGR<->HSV
#函数cv2.cvtColor(input_image,flag)
#flag:cv2.COLOR_BGR2GRAY(灰度), cv2.COLOR_BGR2HSV
'''
img = cv2.imread("soccer.jpg")
#HSV:H(色度)[0,179],S(饱和度)[0,255],V(亮度)[0,255](不同软件对比要归一化)
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.imshow('HSV',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
##########
#物体追踪#
##########

#HSV颜色空间中提取一个特定颜色的物体更容易
import numpy as np
'''
img = cv2.imread('OpenCv_logo.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#设定蓝色阈值
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

#根据阈值构建掩膜,并进行位运算
mask = cv2.inRange(hsv,lower_blue,upper_blue)
res = cv2.bitwise_and(img,img,mask = mask)

res1 = cv2.add(img,res)
cv2.imshow('res',res1)
cv2.imshow('mask',mask)
'''

#如何找到跟踪对象的HSV值

#必须使用三层括号,分别对应cvArray,cvMat,IplImage
#得到转换后绿色的HSV值
def traceHSV_Color(img,color):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #color为BGR模式
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    #取出H,分别用[h-10,100,100]和[h+10,255,255]作为阈值
    h = hsv_color[0,0,0]

    #需要将list转换为numpy.ndarray
    lower_color = np.array([h-20,100,100])
    upper_color = np.array([h+20,255,255])

    mask = cv2.inRange(hsv,lower_color,upper_color)
    res = cv2.bitwise_and(img,img,mask = mask)

    return res
'''
img = cv2.imread('OpenCv_logo.jpg')

green = [0,255,0]
red = [0,0,255]
blue = [255,0,0]

res1 = traceHSV_Color(img,green)
res2 = traceHSV_Color(img,red)
res3 = traceHSV_Color(img,blue)
#add只接受两个src,所以这里加两次
res4 = cv2.add(res1,res2)
res = cv2.add(res4,res3)

cv2.imshow('res',res)
'''
##########
#几何变换#
##########

#OpenCV提供两个变换函数:
#cv2.warpAffine()接受2x3的变换矩阵
#cv2.warpPerspective()接受3x3的变换矩阵
#interpolation(缩放因子):(缩放)cv.INTER_AREA,(扩展)cv2.INTER_CUBIC(慢,效果最佳),cv2.INTER_LINEAR

import numpy as np

#img = cv2.imread('soccer.jpg')

#扩展缩放
#cv2.resize()
#fx,fy和dsize参数不能同时为0,二选一
'''
#此处为设置fx,fy,长宽扩大两倍
res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

#获取dsize参数，为一个元祖
height,width = img.shape[:2]
size = (int(width*0.3),int(height*0.3))

res_dec = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
size = (int(width*2),int(height*2))
res_add = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)

#缩小,放大,原图
cv2.imshow('res_dec',res_dec)
cv2.imshow('res_add',res_add)
cv2.imshow('res_2',res)
cv2.imshow('original',img)
'''
#平移
import matplotlib.pyplot as plt
#还原成RGB
def tran_RGB(img):
    b,g,r = cv2.split(img)
    return cv2.merge([r,g,b])
#img = tran_RGB(img)
'''
#使用np.float32构建移动矩阵,80代表x方向移动,180为y方向移动数值
mat = np.float32([[1,0,80],[0,1,180]])
height,width = img.shape[:2]
size = (width,height)
#warpAffine()接受三个参数,src,移动矩阵,输出大小
res = cv2.warpAffine(img,mat,size)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()
'''
#旋转
#cv2.getRotationMatrix2D()
'''
height,width = img.shape[:2]
#第一个参数为旋转中心,第二个参数为旋转角度
#第三个参数为缩放因子,可用来防止旋转后超出边界的问题
M = cv2.getRotationMatrix2D((width/2,height/2),90,0.6)

size = (int(width*1),int(height*1))
res = cv2.warpAffine(img,M,size)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()
'''
#仿射变换:仿射变换的功能是从二维坐标到二维坐标之间的线性变换,
#且保持二维图形的“平直性”和“平行性”,
#我们需要从源图像中找到三个点以及他们在输出图像中的位置,
#然后由cv2.getAffineTransform()得到变换矩阵
'''
img = cv2.imread('square.jpg')
img = tran_RGB(img)

height,width = img.shape[:2]
size = (width,height)
#变换前后三组点的位置
pst1 = np.float32([[50,50],[200,50],[50,200]])
pst2 = np.float32([[10,100],[200,50],[100,250]])
#得到变换矩阵
M = cv2.getAffineTransform(pst1,pst2)

res = cv2.warpAffine(img,M,size)

plt.subplot(121),plt.imshow(img),plt.title('orginal')
plt.subplot(122),plt.imshow(res),plt.title('after')
plt.show()
'''
#透视变换(三维空间的非线性变换),又称投影映射
#可用于矫正三维图像
#我们需要从源图像中找到四个点以及他们在输出图像中的位置,
#这四个点中的任意三个点都不能共线
#然后由cv2.getPerspectiveTransform()得到变换矩阵
'''
img = cv2.imread('circle.jpg')
img = tran_RGB(img)
height,width = img.shape[:2]

#缩放0.5倍
M = cv2.getRotationMatrix2D((width/2,height/2),0,0.5)
size = (int(width),int(height))
img1 = cv2.warpAffine(img,M,size)
#输入四组点变换前后的位置
pst1 = np.float32([[78,149],[488,149],[283,43],[283,249]])
pst2 = np.float32([[78,149],[488,149],[283,-56],[283,434]])
#得到变换矩阵
M = cv2.getPerspectiveTransform(pst1,pst2)
#使用cv2.warpPerspective()实现透视变换
res = cv2.warpPerspective(img1,M,size)

plt.subplot(121),plt.imshow(img),plt.title('orginal')
plt.subplot(122),plt.imshow(res),plt.title('after')
plt.show()
'''
##########
#图像阈值#
##########

#以灰度模式打开
#img = cv2.imread('OpenCv_logo.jpg',0)

#简单阈值(全局阈值)
#当像素高于阈值,我们给这个像素赋予一个新值,否则赋予另一个新值
'''
#cv2.threshhold()
#这里会返回一个retVal值,在Otsu's二值化会用到
#第一个参数应为灰度图,第二个参数为阈值,第三个参数为高于阈值应当赋予的新值
#第四个参数为不同的阈值法
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original','BINARY','BINARY_INV',"TRUNC","TOZERO","TOZERO_INV"]
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1)
    #显示灰度
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    #将坐标置空
    plt.xticks([]),plt.yticks([])
plt.show()
'''
#自适应阈值:根据图像上每个小区域计算与其对应的阈值
#即不同区域采用不同阈值
#cv2.adaptiveThreshold()
'''
#中值滤波
img = cv2.imread('shudu.jpg',0)
img = cv2.medianBlur(img,5)
#全局阈值
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#cv2.ADAPTIVE_THRESH_MEAN_C:阈值取自相邻区域的平均值
#11是Block size(邻域大小),2是常数(阈值就等于平均值减去这个常数)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                            cv2.THRESH_BINARY,11,2)
#cv2.ADAPTIVE_THRESH_GAUSSIAN_C:取相邻区域的加权和，权重为一个高斯窗口
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                            cv2.THRESH_BINARY,11,2)

titles = ['Original','Global Thresholding(v=127)','Adaptive Mean Thresholding',\
          'Adaptive Gaussian Thresholding']
images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1)
    #显示灰度
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    #将坐标置空
    plt.xticks([]),plt.yticks([])
plt.show()
'''
#Otsu's二值化:对一幅双峰图(直方图中存在两个峰)自动根据其直方图计算出一个阈值
#若不是双峰图,效果不佳
#cv2.threshold()
'''
#同简单阈值相比,增加一个参数cv2.THRESH_OTSU,然后将阈值设置为0
#返回的retVal为最优阈值
img = cv2.imread('noise.jpg',0)
#直接使用二值化
ret1,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#首先使用5x5的高斯核去除噪声,再使用OSTU二值化
blur = cv2.GaussianBlur(img,(5,5),0)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#全局阈值
ret3,th3 = cv2.threshold(img,0,127,cv2.THRESH_BINARY)

images = [img,0,th1,
          img,0,th2,
          blur,0,th3]
titles = ['Original','Histogram','Global Thresholding(v=127)',\
          'Original','Histogram',"Otsu's Thresholding",\
          'Gaussian','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    #这里使用了pyplot中画直方图的方法plt.hist,要注意它的参数是意为数组,
    #所以这里使用了(numpy)ravel方法,将多维数组转换成一维
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
   
plt.show()
'''

############
# 图像平滑 #
############

#2D卷积
#卷积运算：可看作加权求和的过程，使用到的图像区域中的每个像素分别于卷积核(权矩阵)
##########的每个元素相应相乘,所有乘积之和作为区域中心像素的新值.
#卷积核:卷积时使用到的权用一个矩阵表示,该矩阵与使用的图像区域的大小相同，其行,列
########都是奇数,是一个权矩阵.
#对2D图像实施LPF(低通滤波),帮助去除噪音，模糊图像
#实施HPF（高通滤波),帮助我们找到图像的边缘
#cv2.filter2D()

def blur_compare(img):
    
    #建立一个K = 5x5的全1矩阵/25,作为平均滤波器核
    #操作：将核放在图像的一个像素 A 上，求与核对应的图像上 25（5x5）
    #######个像素的和，在取平均数，用这个平均数替代像素 A 的值.
    kernel = np.ones((5,5),np.float32)/25
    #数值为-1,输出数值格式相同的plt.figure()
    dst = cv2.filter2D(img,-1,kernel)

    #图像模糊(平滑)

    #平均:用卷积框覆盖区域所有像素的平均值来代替中心像素
    blur = cv2.blur(img,(5,5))

    #高斯模糊:在平均的基础上,将卷积框中的值变为服从高斯分布,方框中心值最大,
    ##########其余方框根据距离中心原速度距离递减.
    #高斯滤波可以有效的从图像中去除高斯噪音
    #cv2.GaussianBlur()
    #(5,5)是高斯核(奇数),0是X方向的标准差,如果我们只指定了 X 方向的的标准差，
    #Y 方向也会取相同值。如果两个标准差都是 0，那么函数会根据核函数的大小自己计算。
    Gauss_blur = cv2.GaussianBlur(img,(5,5),0)

    #中值模糊:用卷积框对应元素的中值来替代中心像素的值,常用来滤去椒盐噪声
    #5为卷积框尺寸,此处为5x5(image depth?)
    #cv2.medianBlur(img,ksize)
    median = cv2.medianBlur(img,5)

    #双边滤波:在保持边界清晰的情况下有效的去除噪声.
    #同时使用空间高斯权重(高斯模糊)和灰度相似性高斯权重(通过像素的相似度来考虑边界)
    #cv2.bilateralFilter()
    #9是领域直径,两个75分别是空间高斯函数标准差,灰度相似性高斯函数标准差
    bila = cv2.bilateralFilter(img,9,50,75)

    plt.subplot(231),plt.imshow(img),plt.title('Original')
    plt.xticks([]),plt.yticks([])
    plt.subplot(232),plt.imshow(dst),plt.title('Filter2D')
    plt.xticks([]),plt.yticks([])
    plt.subplot(233),plt.imshow(blur),plt.title('Blur')
    plt.xticks([]),plt.yticks([])
    plt.subplot(234),plt.imshow(Gauss_blur),plt.title('Gauss_blur')
    plt.xticks([]),plt.yticks([])
    plt.subplot(235),plt.imshow(median),plt.title('median')
    plt.xticks([]),plt.yticks([])
    plt.subplot(236),plt.imshow(bila),plt.title('bila')
    plt.xticks([]),plt.yticks([])

    plt.show()
'''
img = cv2.imread('OpenCv_logo.jpg')
sault_noise = cv2.imread('sault_noise.jpg')
singer = tran_RGB(cv2.imread('singer.jpg'))
#去除椒盐噪声效果对比
blur_compare(sault_noise)
blur_compare(singer)
'''

############## 形态学操作是根据图像形状进行的简单操作.一般情况下对二值化图像进行的
# 形态学转换 # 操作,需要输入两个参数,一个是原始图像,第二个被称为结构化元素或核,用来
############## 决定操作的性质.
#二值图像：每个像素只有两个可能值的数字图像,常用黑白,单色图像,B&W表示二值图像

#morphology(形态学)
def morphology_tran(img):
    #腐蚀:将前景物体的边界腐蚀掉
    #原理：卷积核沿着图像滑动,如果与卷积核对应的原图像的所有像素值都是1,
    #######那么中心元素就保持原来的像素值，否则就变为零,这将导致
    #######靠近前景的所有像素都会被腐蚀掉(变为0),整幅图像的白色区域会减少
    #对于去除白噪声很有用,也可以用来断开两个连在一块的物体等
    #cv2.erode()
    
    #卷积核5x5
    #iterations迭代次数为1
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations=1)


    #膨胀：与腐蚀原理相反,这个操作会增加图像中的白色区域
    #######一般在去除噪声时先用腐蚀再用膨胀,因腐蚀在去掉白噪声的同时,也会使前景对象
    #######变小所以我们再对他进行膨胀.这时噪声已经被去除了,但是前景还在并会增加.
    #膨胀还可用来连接两个分开的物体
    #cv2.dilate()
    dilation = cv2.dilate(img,kernel,iterations=1)
    
    #以下五种操作皆为同一函数cv2.morphologyEx()

    #开运算：先腐蚀后膨胀,被用于去除噪声
    #参数为cv2.MORPH_OPEN
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #闭运算：先膨胀后腐蚀,常被用来填充前景物体中的小洞或小黑点
    #参数为cv2.MORPH_CLOSE
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    #形态学梯度:其实就是一副图像膨胀与腐蚀的差别,结果看上去就像前景物体的轮廓
    #参数为cv2.MORPH_GRADIENT
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    #礼帽：原始图像与进行开运算后的图像的差
    #参数为cv2.MORPH_TOPHAT
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    #黑帽：原始图像与进行闭运算后的图像的差
    #参数为cv2.MORPH_BLACKHAT
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    plt.subplot(241),plt.imshow(img),plt.title('Original')
    plt.subplot(242),plt.imshow(erosion),plt.title('Erosion')
    plt.subplot(243),plt.imshow(dilation),plt.title('dilation')
    plt.subplot(244),plt.imshow(opening),plt.title('Opening')
    plt.subplot(245),plt.imshow(closing),plt.title('Closing')
    plt.subplot(246),plt.imshow(gradient),plt.title('Gradient')
    plt.subplot(247),plt.imshow(tophat),plt.title('Tophat')
    plt.subplot(248),plt.imshow(blackhat),plt.title('Blackhat')
    plt.savefig('morphology_compare.png')
    plt.show()
'''
img = cv2.imread('lose.jpg')
img2 = cv2.imread('sault_noise.jpg')
morphology_tran(img)
morphology_tran(img2)
'''
#结构化元素:取代前面的numpy,根据提供的形状和大小,自动构建核
#cv2.getStructuringElement()
'''
#正方形的核
kernel_rec = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#椭圆形的核,ellipse(椭圆)
kernel_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#十字形的核
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
'''
############
# 图像梯度 # 梯度简单说就是求导,OpenCv提供三种高通滤波器
############ Sobel,Scharr求一阶导或二阶导数,Laplacian是求二阶导数

#Sobel算子和Scharr算子
#Sobel 算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好.
#你可以设定求导的方向(xorder或yorder).还可以设定卷积核的大小(ksize).如果ksize=-1
#会使用 3x3 的 Scharr 滤波器，它的的效果要比 3x3 的 Sobel 滤波器好,
#而且速度相同，所以在使用 3x3 滤波器时应该尽量使用 Scharr 滤波器
#cv2.Sobel()
#cv2.Scharr()

#Laplacian算子
#可假设其离散实现类似于二阶Sobel导数,事实上,OpenCv在计算时直接调用的Sobel算子.
#cv2.Laplacian()
'''
img = cv2.imread('lose.jpg',0)
#cv2.CV_64F,输出图像的深度,可以使用-1,与原图保持一致
laplacian = cv2.Laplacian(img, -1)
#参数1,0为只在x方向求一阶导数,最大可以求二阶导数
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=5)
#参数0,1为只在y方向求一阶导数,最大可以求二阶导数
sobely = cv2.Sobel(img, -1, 0, 1, ksize=5)

plt.subplot(221),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(laplacian,cmap='gray'),plt.title('Laplacian')
plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(sobelx,cmap='gray'),plt.title('Sobelx')
plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(sobely,cmap='gray'),plt.title('Sobely')
plt.xticks([]),plt.yticks([])
plt.show()
'''
#图像深度:指存储每个像素所用的位数,它确定彩色图像的每个像素可能有的颜色数,
##########或者确定灰度图的每个像素可能有的灰度级数.比如一幅单色图像,若每个象素有8位,
##########则最大灰度数目为2的8次方,即256.
'''
img = cv2.imread('shudu.jpg',0)

sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
#取绝对值后在转换为CV_8U
sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(sobelx8u,cmap='gray'),plt.title('Sobel CV_8U')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(sobel_8u,cmap='gray'),plt.title('Sobel abs(CV_64F)')
plt.xticks([]),plt.yticks([])

plt.show()
'''

################# 
# Canny边缘检测 # Canny 边缘检测是一种非常流行的边缘检测算法,由多步构成
#################

#噪声去除：由于边缘检测很容易受到噪声影响,所以第一步是使用5x5的高斯滤波器去除噪声
#cv2.GaussianBlur()

#计算图像梯度:对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数(图像梯度Gx
############# 和Gy).根据得到的两幅梯度图(Gx,Gy)找到边界的梯度和方向,方向分为四类：
############# 垂直,水平和两个对角线

#非极大值抑制：在获得梯度和方向后,应该对整个图像做一个扫描,去除那些非边界上的点,对每
############## 个像素进行检查,看这个点的梯度是不是周围具有相同梯度方向的点中最大的.

#滞后阈值:现在要确定哪些边界才是真正的边界,这时我们需要设置两个阈值:minVal和maxVal.
######### 当图像的灰度梯度高于maxVal时被认为是真的边界,低于minVal则被抛弃,介于两者
######### 之间的点, 就要看这个点是否与某个被确定为真正的边界点相连,是就认为它也是
######### 边界点,如果不是就抛弃.
#选择适合的阈值非常重要
#在这一步一些小的噪声点也会被除去,因为我们假设边界都是一些长的线段

#cv2.Canny(img,minVal,maxVal,ksize,L2gradient):一个函数完成以上几步
#第二,三个参数为阈值,第四个参数为Sobel卷积核大小(默认为3)
#第五个参数为公式选择(默认False)
def nothing(x):
    pass
#滑块查看阈值对Canny边缘检测算法的影响
def cannySlicebar(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('minVal','image',0,255,nothing)
    cv2.createTrackbar('maxVal','image',0,255,nothing)

    while(1):
        x = cv2.getTrackbarPos('minVal','image')
        y = cv2.getTrackbarPos('maxVal','image')
        edges = cv2.Canny(img,x,y)
        cv2.imshow('image',edges)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

        
#img = cv2.imread('soccer.jpg',0)
#sliceBar(img)
'''
edges_true = cv2.Canny(img,100,200,True)
edges_false = cv2.Canny(img,100,200,False)


plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(edges_true,cmap='gray'),plt.title('Edges_true')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(edges_false,cmap='gray'),plt.title('Edges_false')
plt.xticks([]),plt.yticks([])
plt.show()
'''

############## 同一图像的不同分辨率的原始图像,这组图像就叫做图形金字塔.
# 图像金字塔 # 我们把最大的图像放在底部,最小的放在顶部
############## 有两类金字塔：高斯金字塔和拉普拉斯金字塔

#高斯金字塔：顶部是通过将底部图像中的连续的行和列去除得到的.顶部图像中的每个像素值等于
############ 下一层图像中5个像素的高斯加权平均值.这样操作一次一个(MxN)的图像就变成了一
############ 一个(M/2)x(N/2)的图像,连续进行这样的操作我们就会得到一个分辨率不断下降的
############ 图像金字塔

img = tran_RGB(cv2.imread('Soccer.jpg'))
'''
#cv2.pyrDown()从一个高分辨率大尺寸的图像向上构建一个金字塔(尺寸变小，分辨率降低)
lower_reso = cv2.pyrDown(img)

#cv2.pyrUp()从一个低分辨率小尺寸的图像向下构建一个金字塔(尺寸变大,但分辨率不会增加)
higher_reso = cv2.pyrUp(lower_reso)

plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.subplot(132),plt.imshow(lower_reso),plt.title('lower_reso')
plt.subplot(133),plt.imshow(higher_reso),plt.title('higher_reso')
plt.show()
'''
#图像融合:图像金字塔的一个应用是图像融合. 例如,在图像缝合中,你需要将两幅
######### 图叠在一起，但是由于连接区域图像像素的不连续性,整幅图的效果看起来会
######### 很差. 这时图像金字塔就可以排上用场了,他可以帮你实现无缝连接.

import sys
'''
A = tran_RGB(cv2.imread('apple.jpg'))
B = tran_RGB(cv2.imread('pear.jpg'))

#向上构建A的6层高斯金字塔,存入gpA[]
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
    
#向上构建B的6层高斯金字塔,存入gpB[]
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
#将最小分辨率图像,向下构建5层拉普拉斯金字塔lpA[]
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    #两张图片的长宽必须能够整除2^6,不然此处会报错,两个数组不相等
    #src1使用pyrDown()后,图像的信息就丢失了,再次使用pyrUp()得到的src2虽然大小相同,
    #但无法找回丢失的细节,在这种情况下使用
    #cv2.subtract(src1,src2)图像相减,可以检测两幅图像的差异,从而构建拉普拉斯金字塔.
    L = cv2.subtract(gpA[i-1],GE)
    plt.figure("拉普拉斯金字塔构建")
    plt.subplot(4,5,6-i),plt.imshow(L),plt.title("LpA_{}".format(6-i))
    plt.xticks([]),plt.yticks([])
    lpA.append(L)
    
#向下构建5层拉普拉斯金字塔lpB[]
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    plt.subplot(4,5,11-i),plt.imshow(L),plt.title("LpB_{}".format(6-i))
    plt.xticks([]),plt.yticks([])
    lpB.append(L)
    
LS = []
for la,lb,i in zip(lpA,lpB,range(10,16)):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,:cols//2],lb[:,cols//2:]))
    plt.subplot(4,5,i+1),plt.imshow(ls),plt.title("LpA+B_{}".format(i-9))
    plt.xticks([]),plt.yticks([])
    LS.append(ls)

ls_ = LS[0]

for i in range(1,6):
    plt.subplot(4,5,i+15),plt.imshow(ls_),plt.title("Ls_add_{}".format(i))
    plt.xticks([]),plt.yticks([])
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_,LS[i])
    
#cv2.hstack(tup)堆叠函数,返回一个numpy数组
#此处会将A的左半部分与B的右半部分拼接
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)
plt.show()
'''
#图像金字塔合成图像的方法
def img_pyrmids(A,B):
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
    
    LS = []
    for la,lb,i in zip(lpA,lpB,range(10,16)):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,:cols//2],lb[:,cols//2:]))
        LS.append(ls)

    ls_ = LS[0]

    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_,LS[i])

    return ls_

################## 
# OpenCv中的轮廓 # 轮廓可以简单认为成将连续的点(连着边界)连在一起的曲线,具有想同的
################## 颜色或者灰度. 轮廓在形状分析和物体的检测和识别中很有用.

#初识轮廓

#为了更加准确,要使用二值化图像. 在寻找轮廓之前,要进行阈值化处理或者Canny边界检测.

#查找轮廓的函数会修改原始图像. 如果你在找到轮廓之后还想使用原始图像的话，
#你应该将原始图像存储到其他变量中.
'''
#在 OpenCV 中,查找轮廓就像在黑色背景中超白色物体,要找的物体应该是白色而背景该是黑色.

img = cv2.imread('thunder.jpg')
height,width,dty = img.shape
img = cv2.resize(img,(int(width*0.5),int(height*0.5)),interpolation = 0)
#转为灰度图
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#阈值化
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

#cv2.findContours()查找轮廓
#第二个参数为轮廓检索模式,第三个参数为轮廓近似方法
#返回值第一个为图像,第三个返回值为(轮廓的)层析结构
#第二个为轮廓(一个包含所有轮廓的list,其中的每个轮廓都是一个Numpy数组,
#包含对象边界点的(x,y)坐标)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,\
                                            cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours()
#第二个参数是轮廓,第三个参数是轮廓的索引(设置为-1时,绘制所有轮廓)
#(0,255,0)代表轮廓颜色,后一个3代表轮廓厚度(-1会填充轮廓)
img = cv2.drawContours(img,contours,-1,(0,255,0),3)

#轮廓的近似方法:告诉cv2.findContours()是否所有边界点都需要存储
#cv2.CHAIN_APPROX_NONE(1):存储所有边界点
#cv2.CHAIN_APPROX_SIMPLE(2):去掉冗余点,例如对于一条直线只存储两个端点


#轮廓特征：查找轮廓的不同特征,例如面积,周长,重心,边界框等

##图像矩：可以帮助我们计算图像的质心,面积等?
#cv2.moments(img),以一个字典的形式返回矩
cnt = contours[7]
M = cv2.moments(cnt)

#利用一阶矩计算对象重心C(x,y)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#利用0阶矩计算轮廓的面积,也可以使用cv2.contourArea(contour)
area = M['m00']
#cv2.arcLength()计算轮廓弧长,第二个参数指定图像是闭合(True)还是打开.
perimeter = cv2.arcLength(cnt,True)

#轮廓近似：将轮廓形状近似到另外一种由更少点组成的轮廓形状,新轮廓的点的数目
########## 由我们设定的准确度来决定,使用Douglas-Peucker算法
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

#凸包：与轮廓近似相似,但不同,虽然有些情况下他们的结果是一样的
#凸性缺陷：一般来说凸性曲线总是凸出,至少是平的. 如果有地方凹进去了就称为凸性缺陷
#cv2.convexHull(points,clockwise,returnPoints)
#可以用来检测一个曲线是否具有凸性缺陷,并能纠正缺陷
#参数points为传入的轮廓,clockwise为方向标志(True代表输出的凸包是顺时针方向)
#returnPoints默认为True,它会返回凸包上点的坐标,False则会返回凸包对应轮廓点的索引
hull = cv2.convexHull(cnt)
#凸性检测cv2.isContourConvex(cnt),检验cnt是不是凸的

#边界矩形

##直边界矩形(不考虑对象是否有旋转),所以边界矩形的面积不是最小的
##cv2.boundingRect(),(x,y)为矩形左上角的坐标,(w,h)是矩形的宽和高
x,y,w,h = cv2.boundingRect(cnt)
img_rect = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

##旋转的边界矩形,这个矩形的面积是最小的,因为它考虑了对象的旋转
##cv2.minAreaRect(),返回一个Box2D结构,其中包含左上角点的坐标(x,y),矩形的宽和高(w,h),
##以及旋转角度,但绘制这个矩形需要矩形的四个角点,可通过cv2.boxPoints()获得
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)

#最小外接圆
#cv2.minEnclosingCircle()返回圆心坐标(x,y),半径radius
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,255),2)

#椭圆拟合
#cv2.fitEllipse(),返回旋转边界的内切圆
ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(100,100,255),2)

#直线拟合:根据一组点拟合出一条直线
#cv2.fitLine()
#cv2.DIST_L2为估算器使用的距离,第三个参数为0则选择最佳值
#最后两个0.01,分别为半径精度和角度准确度,通常使用0.01
#返回值(vx,vy)是一个与拟合直线共线的标准化向量,(x0,y0)是拟合直线上的一点
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
#lefty就是截距
lefty = int((-x*vy/vx)+y)
#令y=0,求此时的x值
righty = int((-lefty)*vx/vy)
#分别以(righty,0),(0,lefty)绘图
img = cv2.line(img,(righty,0),(0,lefty),(100,155,200),2)

cv2.imshow('ss',img)
'''

#轮廓的特征
def get_contours(img,thresh=127):
    height,width = img.shape[:2]
    img = cv2.resize(img,(int(width),int(height)),0,interpolation = 0)

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,thresh,255,cv2.THRESH_BINARY)

    image,contours,hier = cv2.findContours(thresh,3,2)
    
    return contours
'''
img = cv2.imread('thunder.jpg')
contours = get_contour(img)
cnt = contours[7]

##长宽比：边界矩形的宽高比
x,y,w,h = cv2.boundingRect(cnt)
aspect_ration = float(w)/h

##Extent:轮廓面积与边界矩形面积的比
area = cv2.contourArea(cnt)
rect_area = w*h
extent = float(area)/rect_area

##Solidity:轮廓面积与凸包面积的比
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area

##Equlvalent Diameter:与轮廓面积相等的圆形的直径
equi_diameter = np.sqrt(4*area/np.pi)

##方向:对象的方向,可使用椭圆拟合求出
(x,y),(Ma,ma),angle = cv2.fitEllipse(cnt)

##掩膜和像素点：有时我们需要构成对象的所有像素点,我们可以这样做
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
#通过寻找掩膜中的不为0的点来获取轮廓对象的所有像素点,返回的是一个数组
pixelpoints = cv2.findNonZero(mask)
#pixelpoints = np.transpose(np.nonzero(mask)),这里的np.transpose为转置
#这里转置的原因是,numpy给出的坐标是(row,colum),而OpenCv则是(x,y)

##最大值和最小值以及它们的位置
##我们可以使用掩膜图来得到这些参数
##cv2.minMaxLoc()
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(imgray,mask = mask)

##平均颜色及平均灰度
##我们也可以用相同的掩膜求一个对象的平均颜色或平均灰度
mean_val = cv2.mean(img,mask=mask)

##极点:一个对象最上面,最下面,最左面,最右面的点
#横轴最小为最左,最大为最右,同理得最上及最下
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
'''

#轮廓：更多函数

'''
##凸缺陷：对象上的任何凹陷都被称为凸缺陷
##cv2.convexityDefect()
##这里returnPoints一定要设置为False,返回轮廓点的索引而不是坐标
##它会返回一个数组,其中每一行包含的值是[起点,终点,最远点,到最远点的近似距离].
##注意这里得到的是点的索引,我们还需要到轮廓点中取找到它们
hull = cv2.convexHull(cnt,returnPoints=False)
defects = cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    #以上面得到的值为索引找轮廓点
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    #将起点与终点连接
    cv2.line(img,start,end,[0,255,0],2)
    #在最远点画圆
    cv2.circle(img,far,5,[0,0,255],-1)

cv2.imshow('img',img)

##Point Polygon Test:求解图像中一个点到一个对象的轮廓的最短距离
##如果点在轮廓外部返回值为负,在轮廓上返回值为0,在轮廓内部返回值为正
##cv2.pointPolygonTest(),第三个参数为True返回最短距离,False只返回位置关系(-1,0,1).
##设置为False,速度会提高2到3倍,若不需要距离,最好设置为False
##以点（50,50)为例
dist = cv2.pointPolygonTest(cnt,(50,50),True)

##形状匹配：比较两个形状或轮廓的相似度,如果返回值越小,匹配越好,它是根据Hu矩来计算的
##Hu矩是归一化中心矩的线性组合,之所以这样做是为了能够获取代表图像的某个特征的矩函数,
##这些矩函数对某些变化如缩放,旋转,镜像映射(除了h1?)具有不变形特性.
##cv2.matchShapes()
imgA = cv2.imread('squareA.jpg')
imgB = cv2.imread('squareB.jpg')
contours_a = get_contours(imgA)
contours_b = get_contours(imgB)
cntA = contours_a[1]
cntB = contours_b[7]
#1为匹配模式,最后一个参数设置为0.0
ret = cv2.matchShapes(cntA,cntB,1,0.0)
'''
'''
#根据点到轮廓的距离绘制不同的颜色
def drawColorByDist(img,cnt):
    height,width = img.shape[:2]
    img = cv2.resize(img,(int(width*0.5),int(height*0.5)),interpolation=cv2.INTER_CUBIC)
    for i in range(int(width*0.5)):
        for j in range(int(height*0.5)):
            dist = cv2.pointPolygonTest(cnt,(i,j),False)
            if dist > 0:
                #注意此处交换了i,j的位置,索引是先列后行
                img[j,i] = [0,255,255]
            elif dist < 0:
                img[j,i] = [255,255,0]
            
            else:
                img[j,i] = [255,0,255]
    return img
img = cv2.imread('thunder.jpg')

contours = get_contours(img)
cnt = contours[7]
img1 = drawColorByDist(img,cnt)
cv2.imshow('img1',img1)
'''
#练习
#查找对应对象的轮廓
def contours_test(img,thresh = 150):
    contours = get_contours(img,thresh)
    for i in range(50):
        img = cv2.drawContours(img,contours,i,(0,255,0),2)
        cv2.imshow('img',img)
        if cv2.waitKey(0) == 99:
            print(i+1)
        elif cv.waitKey(0) == 27:
            break
        else:
            i = i+1
    
#利用cv2.matchShapes()匹配带数字图片,简单识别图片中的数字
def number_recognise(cnt_x):
    img = cv2.imread('numbers.jpg')
    #中值滤波
    img = cv2.medianBlur(img,5)
    contours = get_contours(img,150)
    #以下为第一次运行代码查找每个轮廓对应的数字
    '''
    for i in range(20):
        img = cv2.drawContours(img,contours,i,(0,255,0),2)
        cv2.imshow('img',img)
        if cv2.waitKey(0) == 99:
            print(i)
        else:
            i = i+1
    #根据以上调试得到的0~9的对应轮廓如下
    '''
    cnt7 = contours[1]
    cnt5 = contours[11]
    cnt3 = contours[14]
    cnt2 = contours[15]
    cnt1 = contours[16]
    #同时具有子轮廓,目前没做这部分匹配
    cnt4 = contours[12]#12+13
    cnt0 = contours[2]#2+3
    cnt9 = contours[4]#4+5
    cnt8 = contours[6]#6+7+8
    cnt6 = contours[9]#9+10
    #cnt_0 = list(contours[2])+list(contours[3])
    #cnt0 = np.array(cnt_0)
    ls = [cnt1,cnt2,cnt3,cnt5,cnt7]
    #将每个匹配的结果保存到res1[]
    res1 =[]
    print("match匹配模式3下的轮廓差值：")
    for item in ls:
        #此处参数因数字而异才能正确识别,测试来说3较好
        ret = cv2.matchShapes(cnt_x,item,3,0.0)
        print(ret)
        res1.append(ret)
    #找到最小值对应的ls索引
    res1_c = res1.copy()
    res1_c.sort()
    min_index = res1.index(res1_c[0])
    #判断数字
    if min_index == 0:
        print('数字识别为{}'.format(1))
    elif min_index == 1:
        print('数字识别为{}'.format(2))
    elif min_index == 2:
        print('数字识别为{}'.format(3))
    elif min_index == 3:
        print('数字识别为{}'.format(5))
    elif min_index == 4:
        print('数字识别为{}'.format(7))
    else:
        print('无法识别')
'''
#实验1：数字5,阈值120,cv2.blur(),match参数2,索引3       
img = cv2.imread('number5.jpg')
img = cv2.blur(img,(5,5))
contours = get_contours(img,120)
number_recognise(contours[3])
#contours_test(img,120)

#实验2：数字3,阈值170,cv2.medianBlur(),match参数3,索引11
img = cv2.imread('number3.jpg')
img = cv2.medianBlur(img,5)
contours = get_contours(img,170)
number_recognise(contours[11])
#contours_test(img,170)

#实验3：数字2,阈值170,cv2.meidianBlur(),match参数3,索引101
img = cv2.imread('number2.jpg')
img = cv2.medianBlur(img,5)
contours = get_contours(img,170)
number_recognise(contours[101])
#contours_test(img,170)

#实验4：数字1,阈值150,cv2.blur(),match为3,,索引为0
img = cv2.imread('number100.jpg')
img = cv2.blur(img,(5,5))
contours = get_contours(img,150)
number_recognise(contours[0])
#contours_test(img)

#实验5：数字7,阈值150,cv2.blur(),match参数3,索引7
img = cv2.imread('number7.jpg')
img = cv2.blur(img,(5,5))
contours = get_contours(img,150)
number_recognise(contours[7])
#contours_test(img)
'''

#轮廓的层次结构：探究cv2.findContours()返回的第三个结果,层次结构.一个形状在另外一个
################ 形状的内部,这种情况下我们称外部的形状为父，内部的形状为子.按照这种
################ 方式分类，一幅图像中的所有轮廓之间就建立父子关系。这样我们就可以确
################ 定一个轮廓与其他轮廓是怎样连接的,这种关系就称为组织结构.

##OpenCv中的层次结构,不管层次结构是什么样的,每一个轮廓都包含自己的信息:谁是父,谁是子等,
##OpenCv使用了一个含有四个元素的数组表示——[Next, Previous, First_Child, Parent]
##Next表示同一级组织结构中的下一个轮廓,若没有下一个轮廓,Next = -1.
##Previous同Next,若没有上一个轮廓,Previous = -1.
##First_Child表示它的第一个子轮廓,Parent表示父轮廓,若没有则置为-1.

##轮廓检索模式,cv2.findContours的mode参数

###RETR_LIST：只提取所有轮廓,不创建任何父子关系,所有轮廓处于同一级
###RET_EXTERNAL：只返回最外边的轮廓,所有的子轮廓都会被忽略掉
###RET_CCOMP：在这种模式下会返回所有的轮廓结构并将轮廓分为两级组织结构,一个对象的外轮廓
############# 为一级组织结构,内部的空洞为第二级组织结构,例如数字0,0的外边界属于第一级组织
############# 织结构,0内部的空洞属于第二级组织结构.
###RET_TREE：最完美的一个,这种模式下会返回所有轮廓,并且创建一个完整的组织结构列表.它甚至
############ 能告诉你谁是爸爸,谁是爷爷,儿子,孙子等.
#关于RET_TREE的实例
'''
img = cv2.imread('number100.jpg')

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.blur(imgray,(5,5))
ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
ret,contours,hier = cv2.findContours(thresh,cv2.RETR_TREE,2)

cv2.drawContours(img,contours,-1,(0,255,0),2)
cv2.imshow('img',img)
'''

##########
# 直方图 # 通过直方图你可以对整幅图像的灰度有一个整体的了解,直方图的x轴是灰度值(0~255),
########## y轴是图片中具有同一个灰度值的点的数目

#直方图的计算,绘制与分析

#统计直方图术语
##BINS：如果你不需要知道每个像素点的数目,只需要知道两个像素点之间的像素点数目. 举例来说,
##我们想知道像素值在0到15之间的像素点的数目，接着是16到31,....,240到255. 我们只需要16个值
##来绘制直方图. 于是我们将0~255等分为16组,每一组就被称为BIN,OpenCv中的histSize表示BINS

##DIMS：表示我们收集数据的参数数目,例如若只考虑灰度值,则为1.

##RANGE：就是要统计的灰度值范围,一般来说为[0,256],也就是所有的灰度值

##cv2.calcHist(img,channels,mask,histSize,ranges[,hist[,accumulate]])
##img：传入图像(图像格式为uint8或float32,要使用[]括起来
##channels：要使用中括号括起来,若传入图像为灰度图,则[0]代表灰度图;
########### 若为彩色图像,[0],[1],[2]分别对应通道B,G,R
##mask：统计整幅图像设置为None,但如果你想统计图像某一部分的直方图的话,你就需要制作一个
####### 掩膜图像,然后设置它
##histSize：BIN的数目,也应该用中括号括起来,例如[256]
##ranges：像素值范围,通常为[0,256]
'''
#绘制灰度直方图
img = cv2.imread('summer.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

##使用numpy统计直方图
##np.histogram(),这里的ravel()是将多维数组降至一维
hist1,bins = np.histogram(img.ravel(),256,[0,256])
#hist1和上面一样,bins为256,因为Numpy计算bins的方式为：0-0.99,1-1.99等,所以最后一个数是
#255-255.99,为了表示它,在bins的结尾增加了256
#Numpy还有一个函数np.bincount(),它的运行速度是np.histgram的十倍,所以对于一维直方图,
#最好使用这个np.bincount(img.ravel(),minlength=256)
##注意OpenCv的函数比np.histgram快40倍,所以坚持使用OpenCv的函数

##绘制直方图
#Short Way(简单方法):使用Matplotlib中的绘图函数
#Long Way(复杂方法):使用OpenCV绘图函数

#matplotlib：matplotblib.pyplot.hist()
##它可以直接统计并绘制直方图,plt.hist(img.ravel(),256,[0,256])
##但你应该使用cv2.calcHist()来统计
plt.subplot(131),plt.imshow(img,'gray')
plt.subplot(132),plt.hist(img.ravel(),256,[0,256])
#统计得到的直方图hist通过plt.plot()绘制折线
plt.subplot(133),plt.plot(hist)
plt.show()
'''
'''
##绘制多通道(BGR)的直方图
img = cv2.imread('soccer.jpg')
color = ('b','g','r')
plt.subplot(121),plt.imshow(tran_RGB(img))
plt.subplot(122)

for i,col in enumerate(color):
    hist = cv2.calcHist(img,[i],None,[256],[0,256])
    plt.plot(hist,color=col,linewidth = 1)
    plt.xlim([0,256])
plt.show()
'''
'''
#使用掩膜：构建一副掩膜图像,将要统计的部分设置成白色,其余部分为黑色.
img = cv2.imread('summer.jpg',0)
#构建掩膜
mask = np.zeros(img.shape[:2],np.uint8)
#设置统计部分为白色
mask[200:1200][200:900] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)

hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(121),plt.imshow(masked_img,'gray')
plt.subplot(122),plt.plot(hist_mask)
plt.show()
'''

#直方图均衡化：一副高质量的图像的像素值分布应该很广泛,对于集中在一个像素值范围内
############## 的直方图将它横向拉伸(把比较集中的某个灰度区间变成在全部灰度范围内
############## 的"均匀"分布),这就是直方图均衡化要做的事情,通常情况下这种做法会改
############## 善图像的对比度. 优点技术直观且可逆,缺点是他对处理的数据不加选择,
############## 可能会增加背景杂讯并且降低有用信号的对比度.

#绘制直方图与累积分布图,True表示显示,默认不显示
def draw_hist_cumsumhist(img,a = False):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #hist.cumsum()计算累积和,cdf的第i项就是hist前i项的和
    cdf = hist.cumsum()
    if a == False:
        return cdf
    #标准化,与hist统一
    cdf_normalized = cdf * hist.max()/cdf.max()
    
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.ravel(),256,[0,256],color='r')
    plt.xlim([0,256])
    #legend图例,loc位置
    plt.legend(('cdf','histogram'),loc = 'upper left')
    plt.show()
    return cdf
#直方图均衡化,show为True显示对比图,默认显示
def hist_equalization(img,cdf,show = True):
    #这里导入了numpy中的ma模块,构建了一个numpy的掩膜数组,掩膜数组由一个正常数组和一个
    #布尔数组组成,布尔数组中值为True的元素表示正常数组中对应下标的值无效,False表示有效
    import numpy.ma as ma
    #构建Numpy掩膜数组,cdf为原数组,当数组元素为0时,掩盖(计算时被忽略).
    cdf_m = ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    #对被掩盖的元素赋值,这里赋值为0
    cdf = ma.filled(cdf_m,0).astype('uint8')
    #cdf[img],img的每个元素中的每个值对应cdf中相应的索引,具体参看下例
    '''
    a = np.array([1,2,3])
    b = np.array([[2,1,0],[1,2,0]])
    c = a[b]
    print(c)
    '''
    #上面就获得了一个cdf表,表里面有像素的原像素值对应均衡化后的像素值
    img2 = cdf[img]
    if show == False:
        return img2
    #显示对比图
    plt.subplot(121),plt.imshow(img,'gray'),plt.title('Before')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img2,'gray'),plt.title('After')
    plt.xticks([]),plt.yticks([])
    plt.show()
    return img2
'''
img = cv2.imread('lessLight.jpg',0)
cdf = draw_hist_cumsumhist(img)
img2 = hist_equalization(img,cdf)

##OpenCv中的直方图均衡化,上述过程全部封装在以下函数中
img = cv2.imread('overLight.jpg',0)
img3 = cv2.equalizeHist(img)
cv2.imwrite('equalization_overLight.jpg',img3)
'''

##CLAHE有限对比适应性直方图均衡化
##我们在上边的做的直方图均衡化会改变整个图像的对比度,但很多情况下这样做的效果并不好
##因为很多图像的直方图并不是集中在某一个区域. 为了解决这个问题,我们需要使用自适应的
##直方图均衡化.

##这种情况下, 整幅图像会被分成很多小块(tiles,默认大小是8x8),然后对每一个小块分别进行
##直方图均衡化(和前面类似). 所以直方图会集中在每个区域的某一个小区域中(除非有噪声干扰)
##如果有噪声干扰的话,噪声会被放大,为了避免这种情况的出现要使用对比度限制. 对于每个小块
##来说,如果直方图中的bin超过了对比度的上限的话,就把其中的像素点均匀的分散到其他bins中,
##然后再进行直方图均衡化.
##最后,为了去除每一个小块之间的"人造"(由于算法造成的)边界,再使用双线性插值,对小块进行
##缝合.
##cv2.creatCLAHE()
'''
img = cv2.imread('overLight.jpg',0)
#设置tile大小8x8,clipLimit对比度限制(猜测?)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv2.imshow('img',cl1)
'''

#2D直方图：对于彩色图像,通常情况下我们需考虑两个图像特征,颜色(Hue)和饱和度(Saturation)
'''
##OpenCv中的2D直方图
##首先需要把BGR转换到HSV(同理,计算一维直方图需要将HSV转换为BGR)
img = cv2.imread('colors.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#channels=[0,1]因为我们同时需要处理H和S两个通道
#bins=[180,256]H通道为180,S通道为256
#range=[0,180,0,256]H的取值范围为0~179,S的取值范围在0~255
hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])

##Numpy中的2D直方图
##np.histogram2d()(绘制1维直方图我们使用的是np.histogram())
##第一个参数是H通道,第二个参数是S通道
h = hsv[:,:,0]
s = hsv[:,:,1]
hist_np, xbins, ybins= np.histogram2d(h.ravel(),s.ravel(),[180,256],([0,180],[0,256]))

##绘制2D直方图
##方法1：使用cv2.imshow(),得到一个180x256的二维数组,但这是一个灰度图,除非我们知道
######## 不同颜色 H 通道的值，否则我们根本就不知道那到底代表什么颜色
##方法2：使用plt.imshow(),在使用这个函数时,要记住设置插值参数为nearest.
cv2.imshow('img',hist)
plt.imshow(hist_np,interpolation='nearest')
plt.xlabel('S'),plt.ylabel('H')
plt.show()
'''
#直方图反向投影
##它可以用来做图像分割,或者在图像中找寻我们感兴趣的部分. 简单来说,它会输出与输入图像
##同样大小的图像,其中的每一个像素值代表了输入图像上对应点属于目标对象的概率.用更简单
##的话来解释,输出图像中像素值越高(越白的)的点越可能代表我们要搜索的目标(在输入图像所
##在的位置). 直方图投影经常与camshift算法等一起使用.

##首先我们要为一张包含我们查找目标的图像创建直方图. 我们要查找的对象要尽量占满这张图
##像. 最好使用颜色直方图,因为一个物体的颜色要比它的灰度能更好的被用来进行图像分割与对
##象识别. 接着我们再把这个颜色直方图投影到输入图像中寻找我们的目标,也就是找到输入图像
##中的每一个像素点的像素值在直方图中对应的概率,这样我们就得到一个概率图像,最后设置适当
##的阈值对概率图像进行二值化.
'''
##Numpy中的算法
##首先,我们要创建两幅颜色直方图,目标图像的直方图('M'),输入图像的直方图('I')
roi = cv2.imread('grassland.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

target = cv2.imread('soccer.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

M = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
I = cv2.calcHist([hsvt],[0,1],None,[180,256],[0,180,0,256])
##计算比值R = M/I. 反向投影R,也就是根据R这个"调色板"创建一副新的图像,其中的每一个像素
##代表这个点就是目标的概率.例如B(x,y) = R[h(x,y),s(x,y)],h为(x,y)的Hue值,s为saturation
R = M/I
h,s,v = cv2.split(hsvt)
B = R[h.ravel(),s.ravel()]
##最后加入一个条件B(x,y) = min[B(x,y),1]
B = np.minimum(B,1)
#和hsvt统一格式,array.reshape(height,width)
B = B.reshape(hsvt.shape[:2])
#求一个一个圆盘算子disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#做卷积去除噪音B=BxD,这里D为卷积核disc
B = cv2.filter2D(B,-1,disc)
B = np.uint8(B)
#cv2.normalize()归一化,第三,四个参数为最小,最大值,这里将B中的所有值标准化至0~255范围内
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
ret,B = cv2.threshold(B,30,255,0)
#现在输出图像中灰度值最大的地方就是我们要查找到目标的位置了.如果我们要找的是一个区域,
#使用一个阈值对图像进行二值化,这样就可以得到一个很好的结果了.
plt.subplot(121),plt.imshow(B,'gray')
'''
'''
##OpenCv中的反向投影
##cv2.calcBackProject(),它的参数和cv2.calcHist基本相同
##其中一个参数是我们要查找目标的直方图,同样在使用目标的直方图做反向投影之前,我们应该先
##对其做归一化处理,返回的结果是一个概率图像,我们再使用一个圆盘形卷积核对其做卷积操作,
##最后使用阈值进行二值化.
roi1 = cv2.imread('soccer.jpg')
roi = roi1[500:560,100:200]
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

target = cv2.imread('soccer.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

#获取对象直方图,并归一化
roihist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
#获取目标图像直方图的反向投影,这里第三个参数为对象直方图,最后一个参数scale设置为1
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
#首先getStructuringElement得到椭圆卷积核,然后进行卷积操作
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,disc)
#使用阈值来进行二值化
ret,thresh = cv2.threshold(dst,30,255,0)
#别忘了是三通道图像,这里使用merge变成三通道
thresh = cv2.merge((thresh,thresh,thresh))
#将灰度图,彩色图,原图合并显示
res = cv2.bitwise_and(target,thresh)
res1 = np.hstack((target,thresh,res))

plt.imshow(res1)
plt.show()
'''

############
# 图像变换 #
############

#傅里叶变换：傅里叶变换常被用来分析不同的滤波器的频率特性. 对于对于一个正弦信号,如果它
############ 的幅度变化非常快,我们可以说他是高频信号,如果变化非常慢,我们称之为低频信号. 
############ 你可以把这种想法应用到图像中，图像那里的幅度变化非常大呢？边界点或者噪声.
############ 所以我们说边界和噪声是图像中的高频分量(注意这里的高频是指变化非常快,而非出
############ 现的次数多）。如果没有如此大的幅度变化我们称之为低频分量.

#公式构建频谱图
def get_spectrum(f):
    try:
        ret = f.shape[2]
        spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        return spectrum
    except IndexError:
        magnitude_spectrum = 20*np.log(np.abs(f))
        return magnitude_spectrum
'''
##Numpy中的傅里叶变换
##np.fft.fft2()可以对信号进行转换,输出结果是一个复杂的数组.
##np.fft.ifft2()与以上函数相反,互为可逆操作.
##第一个参数为输入图像,为灰度模式,第二个参数可选,为输出大小
img = cv2.imread('soccer.jpg',0)
f = np.fft.fft2(img)
#现在我们得到了结果,频率为0(0频率也称直流分量)的部分在输出图像的左上角,如果想让它
#在输出图像的中心(这样更容易分析,我们还需要将结果沿两个方向平移N/2,
#函数np.fft.fftshift()可以实现这一点
#函数np.fft.ifftshift()与以上函数相反,可以把直流分量从图像中心移回左上角

fshift = np.fft.fftshift(f)
magnitude_spectrum = get_spectrum(fshift)

fshift[:250,:] = 255
fshift[330:,:] = 255
#构建掩膜,将频谱图中心去掉
#fshift[260:320,380:440] = 0
magnitude_spectrum_back = get_spectrum(fshift)
#将直流分量还原回左上角
f = np.fft.ifftshift(fshift)
#将频谱图还原会正常图像
img_back1 = np.fft.ifft2(f)
#取绝对值
img_back = np.absolute(img_back1)

plt.subplot(221),plt.imshow(img,'gray'),plt.title('Input Image')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum,'gray'),plt.title('Magnitude_spectrum')
plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(img_back,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(magnitude_spectrum_back,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
'''
'''
##OpenCv中的傅里叶变换
##cv2.dft()和cv2.idft()对应上面的np.fft.fft()和np.fft.ifft()
##与前面输出的结果一样,但是是双通道的,第一个通道时结果的实数部分,第二个通道是虚数部分.
img = cv2.imread('soccer.jpg',0)
##输入图像之前要首先转换成np.float32格式.
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
spectrum1 = get_spectrum(dft_shift)

#构建一个60x60的掩膜
rows,cols = img.shape
crow,ccol = rows//2,cols//2

mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift = dft_shift*mask
spectrum1_back = get_spectrum(fshift)

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
#cv2.magnitude()计算二维矢量的幅值,第一个参数为实部,第二个参数为虚部
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


plt.subplot(221),plt.imshow(img,'gray'),plt.title('Input Image')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(spectrum1,'gray'),plt.title('Magnitude_spectrum')
plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.imshow(img_back,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(224),plt.imshow(spectrum1_back,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
'''
#注意OpenCv中的函数cv2.dft()和cv2.idft()要比Numpy快,但Numpy函数对用户更加友好
#DFT的性能优化,当数组大小为某些值时DFT的性能会更好,如果你想提高代码的运行效率,
#你可以修改输入图像的大小(补0),对于OpenCv你必须手动补0,但Numpy在指定大小后自动补0
#那我们怎么确定最佳大小呢?
#OpenCv提供了一个函数cv2.getOptimalDFTSize(),它可以同时被上面两个函数使用.
#使用下列方法,返回修改大小后的图像,以及增加的行数rx,列数cx,性能会提高4倍
def get_optimal_dftsize(img):
    rows,cols = img.shape
    crow = cv2.getOptimalDFTSize(rows)
    ccol = cv2.getOptimalDFTSize(cols)

    rx = crow - rows
    cx = ccol - cols
    
    nimg = np.zeros((crow,ccol),np.uint8)
    nimg[:rows,:cols] = img

    return nimg,rx,cx

##不同算子允许通过的信号
'''
mean_filter = np.ones((3,3))
#高斯滤波
x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T
#scharr in x direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
#sobel in x direction
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
#sobel in y direction
sobel_y = np.array([[-1,-2, 2],
                    [0, 0, 0],
                    [-1, 2, 1]])
#laplacian
laplacian = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]])
filters = [mean_filter, gaussian, scharr, sobel_x, sobel_y, laplacian]
filter_name = ['mean_filter','gaussian', 'scharr', 'sobel_x', 'sobel_y', 'laplacian']

fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(spectrum[i],'gray'),plt.title(filter_name[i])
    plt.xticks([]),plt.yticks([])
plt.show()
#从"各算子允许通过的信号比较.jpg"可以看出mean_filter和gaussian是LPF(低频滤波),
#其余四个全都是是HPF(高频滤波)
'''

############
# 模板匹配 # 模板匹配时用来在一幅大图中搜寻查找模板图像位置的方法
############

'''
#OpenCv中的模板匹配
#cv2.matchTemplate(),和2D卷积一样,它也是用模板图像再输入图像(大图)上滑动,并在每一个位
#置对模板图像和与其对应的输入图像的子区域进行比较,OpenCv提供了几种不同的比较方法(细节
#看文档). 返回的结果是一个灰度图像,每一个像素值表示了此区域与模板的匹配程度.
#如果输入的图像大小是(W,H),模板大小是(w,h),输出的结果大小就是(W-w+1,H-h+1).
#得到输出图像后,就可以使用函数cv2.minMaxLoc()来找到其中的最小值和最大值的位置了.
#第一个值为矩形左上角的点(位置),(w,h)为模板矩形的宽和高,这个矩形就是找到的模板区域了.
img = cv2.imread('messi.jpg',0)
img2 = img.copy()
template = cv2.imread('messi_face.jpg',0)
h,w = template.shape

methods = ['cv2.TM_CCOEFF', ' cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',\
           'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method =eval(meth)
    #进行模板匹配,得到返回的灰度图,每个像素值代表与模板的匹配程度
    res = cv2.matchTemplate(img,template,method)
    #得到最大,最小值及他们的位置
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    #使用不同的比较方法,对结果的解释不同
    #如果使用cv2.TM_SQDIFF或cv2.TM_SQDIFF_NORMED方法,则采用最小值作为左上角的点
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    #否则采用最大值作为左上角的点
    else:
        top_left = max_loc
    #根据左上角的点，以及模板的宽和高,计算右下角的点,
    bottom_right = (top_left[0] + w, top_left[1]+h)
    #以这两个点为基础绘制匹配模板矩形
    cv2.rectangle(img,top_left,bottom_right,255,2)

    plt.subplot(121),plt.imshow(res,'gray'),plt.title('Matching Result')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img,'gray'),plt.title('Detected Point')
    plt.xticks([]),plt.yticks([]),plt.suptitle(meth)

    plt.show()
'''
'''
#多对象的模板匹配：假如你的目标对象在图像中出现了很多次,但函数cv2.minMaxLoc()只会给出
################## 最大值和最小值,此时应该怎么办呢? 使用阈值.
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

template = cv2.imread('mario_coin.jpg',0)
h,w = template.shape

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#设置阈值
threshold = 0.8
#np.where(condition,x,y)根据条件生成新数组,如果只有condition,返回为True元素的位置信息.
#其中x,y参数可选,condition为true返回x,否则返回y.
#这里返回大于阈值的点的位置信息的元组(tuple)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
loc = np.where(res >= 0.8)
#逆序,便于表示先宽后高
loc = loc[::-1]
#loc包含的x,y的位置信息是分散在两个元组的,这里使用zip()将它们合并为一个坐标
for pt in zip(loc[0],loc[1]):
    cv2.rectangle(img_rgb,pt,(pt[0]+ w,pt[1] + h),(0,0,255),1)
cv2.imwrite('mario_coins_found.jpg',img_rgb)
'''

################# 霍夫变换在检测各种形状的技术中非常流行,如果你要检测的形状可以用数学
# Hough直线变换 # 表达式表示出来,你就可以使用霍夫变换检测它.即使要检测的形状存在一点破
################# 坏与扭曲也可以使用.

#原理(略多),上网查
'''
#OpenCv中的霍夫变换
#cv2.HoughLines(),返回值为(ρ,θ). ρ 的单位是像素，θ 的单位是弧度
#第一个参数是一个二值化图像,所以在进行霍夫变换之前首先要进行二值化,或者Canny边缘检测
#第二和第三个值分别代表ρ和θ的精确度,第四个参数是阈值,只有累加其中的值高于阈值时,才
#被认为是一条直线,也可以把它看成能检测到的直线的最短长度(以像素点为单位)
img = cv2.imread('roadLine2.jpg')
img2 = img.copy()
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#这里可以使用前面定义的cannySlicebar()最佳min_val和max_val
cannySlicebar(img_gray)
edges = cv2.Canny(img_gray,50,150,3)

lines = cv2.HoughLines(edges,1,np.pi/180,100)
for line in lines:
    for rho,theta in line:
        #以下根据rho,theta求直线
        a = np.cos(theta)
        b = np.sin(theta)
        
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('img1',img)

#Probalbilistic Hough Transform
#从上面我们可以发现:仅仅是一条之心啊都需要两个参数,这需要大量的计算,而PHT是对霍夫变换的
#一种优化. 它不会对每一个点都进行计算,而是从一幅图中随机选取(是不是也可以用图像金字塔呢)
#一个点集进行计算,对于直线检测来说这已经足够了,但是这种变换我们呢必须要降低阈值(因为总
#像素点少了).
#cv2.HoughLinesP(),它增加了两个参数
#minLineLength：线的最短长度,比这个短的线都会被忽略
#maxLineGap:两条线段之间的最大间隔,如果小于此值,这两条线就被看成一条直线
#更加给力的是,这个函数的返回值就是直线的起点和终点,一切变得跟家直接和简单
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('img2',img2)
'''

################
# 霍夫圆环变换 # 使用霍夫变换在图像中找圆形(环)
################
'''
#原理：圆形的数学表达式为 (x − xcenter)^2+(y − ycenter)^2 = r^2，其中（xcenter,ycenter）
###### 为圆心的坐标,r为圆的半径.从这个等式我们可以看出：一个圆环需要3个参数来确定. 所以
###### 进行圆环霍夫变换的累加器必须是3维的,这样的话效率就会很低,所以OpenCv使用了一个比较
###### 巧妙的方法,霍夫梯度法,它可以使用边界的梯度信息.

#cv2.HoughCircles(),返回值为一个包含(x,y,radius)的序列
#method：目前只能使用CV_HOUGH_GRADIENT
#dp:累加器分辨率与图像分辨率的倒数比,若dp=2,累加器分辨率为图像分辨率的一半
#minDist：圆心之间的最小距离
#param1,param2:分别为Canny()中的max_val和检测阶段的圆心累加器阈值
#minRadius,maxRadius:检测到的最小圆半径,最大圆半径
img = cv2.imread('eyeball.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#建立圆形掩膜,缩小处理区域
mask = np.zeros(img_gray.shape,np.uint8)
cv2.circle(mask,(235,158),100,(255,255,255),-1)
img1 = cv2.add(img_gray,img_gray,mask = mask)

circles = cv2.HoughCircles(img1,cv2.HOUGH_GRADIENT,1,40,param1=131,param2=43,\
                           minRadius=50,maxRadius=83)
for circle in circles:
    for x,y,r in circle:
        cv2.circle(img,(x,y),r,(0,255,0),2)
cv2.imwrite('eyeball.circle.jpg',img)
'''

######################
# 分水岭算法图像分割 #
######################

#原理:任何一副灰度图像都可以被看成拓扑平面, 灰度值高的区域可以被看成是山峰, 灰度值低
##### 的区域可以被看成是山谷. 我们向每一个山谷中灌不同颜色的水. 随着水的位的升高,不
##### 同山谷的水就会相遇汇合, 为了防止不同山谷的水汇合, 我们需要在水汇合的地方构建起
##### 堤坝. 不停的灌水,不停的构建堤坝知道所有的山峰都被水淹没. 我们构建好的堤坝就是
##### 对图像的分割. 这就是分水岭算法的背后哲理

#但是这种方法通常都会得到过度分割的结果,这时由于噪声或者图像中其他不规律的因素造成的.
#为了减少这种影响,OpenCv采用了基于掩膜的分水岭算法,在这种算法中我们要设置哪些山谷点汇合
#哪些不会. 这时一种交互式的图像分割,我们要做的就是给我们已知的对象打上不同的标签.如果某
#个区域肯定是前景或对象,就使用某个颜色(或灰度值)标签标记它. 如果某个区域肯定不是对象而
#是背景就使用另外一个颜色标签标记,而剩下的不能确定是前景还是背景的区域就用0标记.这就是
#我们的标签. 然后实施分水岭算法,每一次灌水,我们的标签就会被更新,当两个不同颜色的标签相
#遇时就构建堤坝,直到所有的山峰淹没,最后我们得到的边界对象(堤坝)的值为-1.
'''
#图像二值化
img = cv2.imread('water_coins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('img',thresh)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel,iterations=1)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
#这里采用了距离转换算法,首先使用cv2.distanceTransform()
#它计算每个二值图像像素到最近的0像素的近似距离或精确距离,对于0图像像素,距离显然为0.
#第二个参数为distanceType,这里选cv2.DIST_L1
#第三个参数为maskSize,距离变换蒙版大小,它可以是3,5等
dist_transform = cv2.distanceTransform(opening,1,5)
#然后应用一个阈值来决定哪些区域是前景
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
#前景后景相减来找到未知区域
unknown = cv2.subtract(sure_bg,sure_fg)
#cv2.connectedComponents()会将背景标记为0,其他的对象使用从1开始的正整数标记.
ret,markers1 = cv2.connectedComponents(sure_fg)
#但如果背景标记为0,那么分水岭算法就会把它当成未知区域了,所以我们想使用不同的整数标记它们
markers = markers1+1
#而对不确定区域,使用unknown标记为0
markers[unknown==255] = 0
#cv2.watershed(),分水岭函数,第二个参数为标记地图,其大小与输入图像相同,函数返回标记地图.
markers3 = cv2.watershed(img,markers)
#-1代表分水岭边界值,其余的正数值代表对象的编号
img[markers3 == -1] = [255,0,0]
cv2.imshow('img4',img)
'''

#####################################
# 使用GrabCut算法进行交互式前景提取 # 此算法在提取前景的操作过程需要很少的人机交互,
##################################### 结果非常好

#原理：用户角度,开始时用户需要用一个矩形将前景区域框住(前景区域应该完全包括在矩形框内部)
###### 然后算法进行迭代式分割,直到达到最好结果. 但是有时分割的结果不够理想,比如把前景当
###### 成了背景,或者把背景当成了前景. 在这种情况下,就需要用户来进行修改了.用户只需要在
###### 不理想的部位画一笔(点一下鼠标)就可以了,画一笔就等于在告诉计算机："嗨,老兄,你把这
###### 里弄反了,下次迭代的时候记得改过来呀!". 然后下一轮迭代的时候你就会得到一个更好的
###### 结果了

#程序角度：

#用户输入一个矩形。矩形外的所有区域肯定都是背景（我们在前面已经提到，所有的对象都要包
##含在矩形框内).计算机会对我们的输入图像做一个初始化标记。它会标记前景和背景像素.


#使用一个高斯混合模型(GMM)对前景和背景建模
##根据我们的输入，GMM 会学习并创建新的像素分布。对那些分类未知的像素（可能是前景也可能
##是背景），可以根据它们与已知分类（如背景）的像素的关系来进行分类(就像是在做聚类操作)
##这样就会根据像素的分布创建一副图。图中的节点就是像素点。除了像素点做节点之外还有两个
##节点：Source_node和Sink_node。所有的前景像素都和Source_node相连.所有的背景像素都和
##Sink_node 相连。


#将像素连接到 Source_node/end_node（边）的权重由它们属于同一类（同是前景或同是背景）
##的概率来决定。两个像素之间的权重由边的信息或者两个像素的相似性来决定。如果两个像素的
##颜色有很大的不同，那么它们之间的边的权重就会很小。


#使用 mincut 算法对上面得到的图进行分割
##它会根据最低成本方程将图分为 Source_node 和 Sink_node。成本方程就是被剪掉的所有边的权
##重之和。在裁剪之后，所有连接到 Source_node 的像素被认为是前景，所有连接到 Sink_node
##的像素被认为是背景

#继续这个过程直到分类收敛

#cv2.grabCut()
#mask：掩膜图像,用来确定哪些区域是背景,前景,可能是前景/背景等. 可以设置为cv2.GC_BGD,
###### cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD,分别对应0,1,2,3
#rect:包含前景的矩形,格式为(x,y,w,h)
#bdgModel,fgdModel：算法内部使用的数组,你只需要创建两个大小为(1,65),数据类型为
################### np.float64的数组
#iterCount:算法迭代次数
#mode可以设置为cv2.GC_INIT_WITH_RECT或cv2.GC_INIT_WITH_MASK. 也可以联合使用,这是用来确
#定我们的修改模式,矩形模式或者掩膜模式

#交互式grabCut实例
'''
#设置鼠标回调函数
def drawCut(event,x,y,flags,param):
    global draw,mode,x1,y1,x0,y0
    if event == cv2.EVENT_LBUTTONDOWN and mode == 0:
        x0,y0 = x,y
        draw = True
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if mode == 1:
            cv2.circle(mask_mannual,(x,y),1,255,2)
            cv2.circle(img2,(x,y),1,(0,255,0),2)
            print('mask1修改')
        elif mode == 2:
            cv2.circle(mask_mannual,(x,y),2,1,2)
            cv2.circle(img2,(x,y),1,(0,0,255),2)
            print('mask2修改')
    elif event == cv2.EVENT_LBUTTONUP and draw == True and mode == 0:
        x1,y1 = x,y
        cv2.rectangle(img2,(x0,y0),(x1,y1),(0,255,0),1)
        draw = False

img = cv2.imread('messi.jpg')
#分别建立初始掩膜,人为绘制掩膜,和恢复掩膜
mask = np.zeros(img.shape[:2],np.uint8)
mask_mannual = mask.copy()
mask_reset = mask.copy()
#创建前后景模型
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
draw = False
#0为矩形,1为直线画笔
mode = 0
cv2.namedWindow('image')
#设置矩形初始变量
x0,y0,x1,y1 = 0,0,0,0
img2 = img.copy()
cv2.setMouseCallback('image',drawCut)
print('############################\n\
##grabCut交互式图像分割示例#\n\
############################\n')
print('请先划定前景的大致矩形区域')
while(1):
    cv2.imshow('image',img2)
    choice = cv2.waitKey(0)
    print('c--------分割')
    print('Esc------退出')
    #'c',cut图像
    if choice == 99:
        print('开始分割')
        #创建矩形
        rect = (x0,y0,abs(x1-x0),abs(y1-y0))
        #函数的返回值是更新的mask,bgdModel,fgdModel,这里没有返回值,但mask已经改变
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        #astype转换格式
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        #np.newaxis为多维数组增加1维
        img2 = img*mask2[:,:,np.newaxis]
        #显示初步切割的图像
        cv2.imshow('image',img2)
        print('m------修正')
        choice1 = cv2.waitKey(0)
        #'m'切换模式
        if choice1 == 109:
            mode = 1
            print('开始进一步分割')
            
            choice2 = 0
            #'q'或者esc退出循环
            count = 1
            while(count):
                print('请使用画笔修改,mode 1 为增加,2 为去除\n'+
                      '当前mode {}'.format(mode))
                cv2.imshow('image',img2)
               
                print('q--------退出修正')
                print('d--------修正完成')
                #获取当前mode
                choice2 = cv2.waitKey(0)
                
                if choice2 == 109:
                    if mode == 1:
                        mode =2
                    else:
                        mode = 1
                
                #'d'完成画图,开始修改
                if choice2 == 100:
                    print('修正完成')
                    mask[mask_mannual == 255] = 1
                    mask[mask_mannual == 1] = 0
                    #mask[mask == 0] = 0
                    ret = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,\
                                      cv2.GC_INIT_WITH_MASK)
                    
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    
                    img2 = img*mask2[:,:,np.newaxis]
                    cv2.imshow('image',img2)
                print('s--------保存图像')
                if choice2 ==115:
                    cv2.imwrite('Cut.jpg',img2)
                    print('保存成功')
                if choice2 == 113 :
                    count = 0
                    mode = 0
                    img2 = img.copy()
                    mask = mask_reset
                    print('请划定前景的大致矩形区域')
        else:
            pass
    #esc退出
    elif choice == 27:
        break
        
cv2.destroyAllWindows()
'''

######################
# 图像特征提取与描述 #
######################

################ 
# 理解图像特征 # 特征需要具有唯一性,适于被跟踪,容易被比较
################

#找到图像特征的技术被称为'特征检测'技术. 对特征周围的区域进行描述,这样他才能在其他图像
#找到相同的特征,我们把这种描述称为'特征描述'.当你有了特征描述后,你就可以在所有的图像中
#找这个相同的特征了.

##################
# Harris角点检测 # 我们已经知道了角点的一个特性：向任何方向移动,变化都很大.
##################

#原理:主要是公式,不好记录,上网查

#OpenCv中的Harris角点检测
#cv2.cornerHarris()
#img：数据类型为float32的输入图像
#blockSize：角点检测中要考虑的领域大小
#ksize：Sobel求导中使用的窗口大小
#k：Harris角点检测方程中的自由参数,取值参数为[0.04,0.06]
'''
img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#必须转换为float32格式
gray = np.float32(gray)
#得到每个点为角点的分数
dst = cv2.cornerHarris(gray,2,3,0.04)
#膨胀操作使角点更加直观,不重要
dst = cv2.dilate(dst,None)
#设置阈值,这个阈值和图像关系非常大
img[dst>0.01*dst.max()] = [0,255,0]
cv2.imshow('dst',img)

#亚像素级精确度的角点:有时我们需要最大精确度的角点检测
#cv2.cornerSubPix()
ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#centroids质心
ret,labels,stats,centroids = cv2.connectedComponentsWithStats(dst)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

res = np.hstack((centroids,corners))
#np.int0,可以用来省略小数点后面的数字(非四舍五入)
res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,2]

cv2.imwrite('subpixel5.jpg',img)
'''
#############################################
# Shi_Tomasi角点检测 & 适合于跟踪的图像特征 # Harris算法的改进,只有当λ1和λ2都大于
############################################# 阈值,才被认为是角点

#cv2.goodFeaturesToTrack(),这个函数可以帮我们用Shi_Tomasi方法获取图像中的N个最好的角点
#输入的应该是灰度图像,然后确定你想要检测到的角点数目,再设置角点的质量水平(0到1之间). 它
#代表了角点的最低质量,低于这个数的所有角点都会被忽略,最后设置两个角点之间的最短欧氏距离

#根据这些信息,函数就能在图像上找到角点. 所有低于质量水平的角点都会被忽略,然后再把合格角
#点按角点质量进行降序排列. 函数会采用角点质量最高的那个角点(排序后的第一个),然后将它附
#近(最小距离之内)的角点都删掉,按着这样的方式最后返回N个最佳角点.
'''
img = cv2.imread('buildingblocks.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,40,0.01,10)
corners = np.int0(corners)
#绘图
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()
'''

###############################################
# 介绍SIFT(Scale-Invariant Feature Transform) # 尺度不变特征转换
###############################################

#原理：前面的Harris以及Shi_Tomasi角点检测具有旋转不变特性,即使图片发生了旋转,我们也能
###### 找到同样的角点. 很明显即使图像发生旋转之后角点还是角点. 那如果我们对图像进行缩
###### 放呢？ 角点可能就不是角点了,所以在2004年,D.Lowe提出了一个新的算法：尺度不变特征
###### 转换(SIFT),这个算法可以帮助我们提取图像中的关键点并计算它们的描述符. 共分四步.

#尺度空间极值检测

##不同的尺度空间不能使用相同的窗口检测极值点. 对小的角点要用小的窗口,对大的角点只能使用
##大的窗口. 为了达到这个目的我们要使用 尺度空间滤波器(尺度空间滤波器可以使用一些具有不
##同方差 σ 的高斯卷积核构成. 使用具有不同方差值 σ 的高斯拉普拉斯算子（LoG）对图像进行
##卷积，LoG 由于具有不同的方差值 σ 所以可以用来检测不同大小的斑点（当 LoG 的方差 σ 与
##斑点直径相等时能够使斑点完全平滑）。简单来说方差 σ 就是一个尺度变换因子,（高斯方差
##的大小与窗口的大小存在一个倍数关系：窗口大小等于 6 倍方差加 1，所以方差的大小也决定了
##窗口大小.

##但是这个LoG的计算量非常大,所以SIFT算法使用高斯差分算子(DoG)来对LoG做近似
##这里需要再解释一下图像金字塔，我们可以通过减少采样（如只取奇数行或奇数列）来构成一组
##图像尺寸（1，0.5，0.25 等）不同的金字塔，然后对这一组图像中的每一张图像使用具有不同
##方差 σ 的高斯卷积核构建出具有不同分辨率的图像金字塔（不同的尺度空间). DoG 就是这组
##具有不同分辨率的图像金字塔中相邻的两层之间的差值.

##在DoG搞定之后,就可以在不同的尺度空间和2D平面中搜索局部最大值了. 对于图像中的一个像素
##点而言,它需要与自己周围的 8领域 ,以及尺度空间中上下两层中的相邻的 18(2x9)个点相比.如
##果是局部最大值,它就可能是一个关键点. 基本上来说关键点是图像在相应尺度空间中的最好代表
##该算法的作者在文章中给出了SIFT参数的经验值：octaves=4(4层图像金字塔),尺度空间为5,也就
##是每个尺寸使用5个不同方差的高斯核进行卷积,初始方差1.6,k=√2等

#关键点(极值点)定位

##一旦找到关键点,我们就要对它们进行修正从而得到更准确的结果.作者使用尺度空间的泰勒级数
##展开,如果极值点的灰度值小于阈值(0.03)就会被忽略掉.
##在OpenCv中称这类阈值contrastThreshold
##DoG算法对边界非常敏感,所以我们必须把边界去除. 前面我们讲的Harris 算法除了可以用于角
##点检测之外还可以用于检测边界。从 Harris 角点检测的算法中，我们知道当一个特征值远远大
##于另外一个特征值时检测到的是边界。所以他们使用了一个简单的函数，如果比例高于阈值
##（OpenCV 中称为边界阈值），这个关键点就会被忽略. 所以低对比度的关键点和边界关键点都会
##被去除掉,剩下的就是我们感兴趣的关键点了.

#为关键点(极值点)指定方向参数

##现在我们要为每一个关键点赋予一个方向参数，这样它才会具有旋转不变性. 获取关键点所在尺
##度空间的领域,然后计算这个区域的梯度级和方向. 根据得到的结果创建一个含有36个bins（每
##10度一个bin）的方向直方图(使用当前尺度空间 σ 值的 1.5 倍为方差的圆形高斯窗口和梯度
##级做权重). 直方图中的峰值为主方向参数,如果其他的任何柱子的高度高于峰值的80%被认为是辅
##方向. 这就会在相同的尺度空间,相同的位置构建出具有不同方向的关键点,这对于匹配的稳定性
##会有所帮助.

#关键点描述符

##新的关键点描述符被创建了。选取与关键点周围一个 16x16 的邻域，把它分成 16 个 4x4 的小
##方块，为每个小方块创建一个具有 8 个 bin 的方向直方图。总共加起来有 128 个 bin。由此组
##成长为 128 的向量就构成了关键点描述符。除此之外还要进行几个测量以达到对光照变化，旋转
##等的稳定性。

#关键点匹配

##下一步就可以采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量.取第一
##个图的某个关键点，通过遍历找到第二幅图像中的距离最近的那个关键点. 但有些情况下，第二
##个距离最近的关键点与第一个距离最近的关键点靠的太近。这可能是由于噪声等引起的。此时要
##计算最近距离与第二近距离的比值。如果比值大于 0.8，就忽略掉。这会去除 90% 的错误匹配,
##同时只去除 5% 的正确匹配。
'''
#OpenCv中的SIFT
img = cv2.imread('church.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#SIFT算法已经被整合到模块opencv_contrib中,需另外安装
sift = cv2.SIFT()
#sift.detect函数可以在图像中查找关键点. 如果你只想在图像中的一个区域搜索的话,也可以创建
#一个掩码图像作为参数使用,返回的关键点kp是一个带有很多不同属性的特殊结构体,这些属性中包
#含它的坐标(x,y),有意义的邻域大小,确定其方向的角度等
kp = sift.detect(gray,None)
#cv2.drawKeyPoints(),OpenCv提供的绘制关键点的函数,它可以在关键点的部位绘制一个小圆圈,如
#果你设置参数为
img = cv2.drawKeypoints(gray,kp)

cv2.imshow('sift_keypoints.jpg',img)

########################################
# 介绍SURF(Speeded-Up Robust Features) # 加速稳健特征,对SIFT算法的改进
########################################

#原理：在 SIFT 中，Lowe 在构建尺度空间时使用 DoG 对 LoG 进行近似。SURF
###### 使用盒子滤波器（box_filter）对 LoG 进行近似. 在进行卷积计算时可以利用积分图像
######（积分图像的一大特点是：计算图像中某个窗口内所有像素和时，计算量的大小与窗口大小无
###### 关），是盒子滤波器的一大优点。而且这种计算可以在不同尺度空间同时进行。同样 SURF
###### 算法计算关键点的尺度和位置是也是依赖于Hessian矩阵行列式的。

#为了保证特征矢量具有旋转不变形,需要对每一个特征点分配一个主要方向. 需要以特征点为中心,
#以6s(s为特征点的尺度)为半径的圆形区域内,对图像进行Harr小波相应运. 这样做实际就是对图
#像进行梯度运算,但是利用积分图像,可以提高计算图像梯度的效率,为了求取主方向值,需要设计
#一个以方向为中心,张角为60度的扇形滑动窗口,以步长为0.2弧度左右旋转这个滑动窗口,并对窗
#口内的图像Haar小波的响应值进行累加. 主方向为最大的Haar响应累加值的方向. 在很多应用中
#根本不需要旋转不变性,所以没有必要确定它们的方向,如果不计算方向的话,又可以使算法提速.
#SURF提供了成为U-SURF的功能,它具有更快的速度,同时又保持了对+/-15度旋转的稳定性. OpenCv
#对两种模式同时支持,只需要对参数upright进行设置,当upright为0时计算方向,为1时不计算方向,
#同时速度更快。

#注意对于SURF和SIFT算法,因未装好opencv-contrib库,暂不贴代码
'''

###################### 前面的特征检测器虽然效果很好,但是从实时处理的角度来看,这些算法
# 角点检测的FAST算法 # 都不够快,一个最好的例子就是SLAM(同步定位与地图构建),移动机器人
###################### 它们的资源非常有限. 06年提出的FAST算法解决了这个问题

#使用FAST算法进行特征提取
#1.在图像中选取一个像素点p,在图像中选取一个像素点p,来判断它是不是关键点. Ip 等于像素点
###p的灰度值.
#2.选择适当的阈值t
#3.在像素点p的周围选择16个像素点进行测试
#4.如果在这 16 个像素点中存在 n 个连续像素点的灰度值都高于Ip+t,或者低于Ip−t,那么像素点
###p 就被认为是一个角点.
#5.为了获得更快的效果,还采用了而外的加速办法。首先对候选点的周围每个90度的点(编号)：1,
### 9,5,13 进行测试（先测试1和9如果它们符合阈值要求再测试 5 和13）. 如果p是角点,那么这
###四个点中至少有 3 个要符合阈值要求。如果不是的话肯定不是角点，就放弃。对通过这步测试
###的点再继续进行测试.

#这个检测器效率很高,但是它有如下几条缺点：
##当 n<12 时它不会丢弃很多候选点 (获得的候选点比较多)
##像素的选取不是最优的，因为它的效果取决与要解决的问题和角点的分布情况
##高速测试的结果被抛弃
##检测到的很多特征点都是连在一起的

#前三个问题可以通过机器学习的方法解决,最后一个问题可以使用非最大值抑制的方法解决

#机器学习的角点检测
#1. 选择一组训练图片（最好是跟最后应用相关的图片）
#2. 使用 FAST 算法找出每幅图像的特征点
#3. 对每一个特征点,将其周围的16个像素存储构成一个向量.对所有图像都这样做构建一个特征
####向量 P.
#4. 每一个特征点的 16 像素点都属于下列三类中的一种:d[,Ip-t] ,s[Ip-t,Ip+t] ,b[Ip+t]
#5. 根据这些像素点的分类，特征向量 P 也被分为 3 个子集：Pd，Ps，Pb

#6. 定义一个新的布尔变量 Kp，如果 p 是角点就设置为 Ture，如果不是就设置为 False
#7. 使用ID3算法(决策树分类器). 熵的定义为每个类别的概率乘以该类别概率的对数,然后求和.
####根据定义得知,如果每个类别的概率分布很均匀,则熵越大,例如假如每个类别的概率都是1/n,
####则熵取最大值,越均匀越不稳定,如果某一个类别的概率取1,其余类别的概率为0,则熵取最小值
####我们希望在利用某个特征划分集合后,两个子集的熵越小越好,越小说明某一个类别占得比例很
####大,其余类别很小.
#8. 对所有子集进行递归,直到它们的熵为0
#9. 将构建好的决策树运用于其他图像的快速的检测.

#非极大值抑制(NMS）
#使用非极大值抑制的方法可以解决检测到的特征点相连的问题
#1. 对所有检测到到特征点构建一个打分函数 V。V 就是像素点 p 与周围 16个像素点差值的绝对
####值之和
#2. 计算临近两个特征点的打分函数 V.
#3. 忽略V值最低的特征点

#总结：FAST算法比其他角点检测算法都快,但是在噪声很高时不够稳定,这是由阈值决定的.

#OpenCv中的FAST特征检测器
#如果你愿意的话,你还可以设置阈值,是否进行非最大值抑制,要使用的邻域大小等
#邻域设置为下列几种之一：cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
#cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,cv2.FAST_FEATURE_DETECTOR_TYPE_7_12
'''
img = cv2.imread('buildingblocks.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fast=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True,\
                                    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

kp = fast.detect(gray,None)
#第三个参数为输出图像
img2 = cv2.drawKeypoints(img,kp,img)
#打印所有默认参数
#阈值
print('Threshold:', fast.getThreshold())
#是否采取非极大值抑制
print('nonmaxSuppression:',fast.getNonmaxSuppression())

print('neighborhood:',fast.getType())
#采取非极大值抑制下的边界点数目
print('Total Keypoints with nonmaxSuppression:',len(kp))

cv2.imwrite('fast_true.jpg',img2)
#设置非抑制极大值参数为0
fast.setNonmaxSuppression(0)
kp = fast.detect(gray,None)
#未采取非极大值抑制下的边界点数目
print('Total Keypoints without nonmaxSuppression：',len(kp))

img3 = cv2.drawKeypoints(img, kp,img)

cv2.imwrite('fast_false.jpg',img3)
'''

##########################################################
# BRIEF( Binary Robust Independent Elementary Features ) #
##########################################################

#我们知道SIFT算法使用的是128维的描述符。由于它是使用的浮点数，所以要使用 512 个字节
#同样 SURF 算法最少使用 256 个字节（64 为维描述符）。创建一个包含上千个特征的向量需
#要消耗大量的内存，在嵌入式等资源有限的设备上这样是合适的。匹配时还会消耗更多的内存
#和时间。但是在实际的匹配过程中如此多的维度是没有必要的。我们可以使用 PCA，LDA 等方
#法来进行降维。甚至可以使用 LSH（局部敏感哈希）将 SIFT 浮点数的描述符转换成二进制字
#符串。对这些字符串再使用汉明距离进行匹配。汉明距离的计算只需要进行 XOR 位运算以及位
#计数，这种计算很适合在现代的CPU 上进行。但我们还是要先找到描述符才能使用哈希，这不
#能解决最初的内存消耗问题。

#BRIEF应运而生。它不去计算描述符而是直接找到一个二进制字符串. 这种算法使用的是已经平
#滑后的图像,它会按照一种特定的方式选取一组像素点对nd (x，y),然后在这些像素点对之间进
#行灰度值对比.例如，第一个点对的灰度值分别为 p 和 q。如果 p 小于 q，结果就是 1，否则
#就是 0。就这样对 nd个点对进行对比得到一个 nd 维的二进制字符串. nd 可以是 128，256，
#512。OpenCV 对这些都提供了支持，但在默认情况下是 256（OpenC 是使用字节表示它们的，所
#以这些值分别对应与 16，32，64）。当我们获得这些二进制字符串之后就可以使用汉明距离对
#它们进行匹配了。


#非常重要的一点是：BRIEF是一种特征描述符,它不提供查找特征的方法,所以我们不得不使用其他
#的特征检测器. 比如SIFT和SURF等,原始文献推荐时用CenSurE特征检测器,这种算法很快。而且
#BRIEF 算法对 CenSurE关键点的描述效果要比 SURF 关键点的描述更好,简单来说BRIEF是一种对
#特征点描述符计算和匹配的快速方法. 这种算法可以实现很高的识别率,除非出现平面内的大旋转.
'''
#OpenCv中的BRIEF
#下面使用了CenSurE特征检测器和BRIEF描述符(在OpenCv中CenSurE检测器被叫做STAR检测器)
img = cv2.('buildingblocks.jpg',0)
#这里仍需要加载opencv-contrib库,在shell中无法加载,以下代码使用Anaconda调试
#初始化Star检测器
star = cv2.xfeatures2d.StarDetector_create()
#初始化BRIEF提取器
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#找出关键点
kp = star.detect(img,None)
#使用BRIEF得到关键点的特征描述
kp,des = brief.compute(img,kp)

print(brief.getInt('byte'))
print(des.shape)
'''

########################################
# ORB( Oriented FAST and Rotated BRIEF # SIFT和SURF算法的一个很好的替代品,比起前面 
######################################## 两个算法快很多

#原理：ORB 基本是 FAST 关键点检测和 BRIEF 关键点描述器的结合体，并通过很多修改增强了
#######性能。首先它使用 FAST 找到关键点，然后再使用 Harris角点检测对这些关键点进行排序
#######找到其中的前 N 个点。它也使用金字塔从而产生尺度不变性特征。

#但是有一个问题，FAST算法不计算方向，那旋转不变性怎样解决呢？
#它使用灰度矩的算法计算出角点的方向。以角点到角点所在（小块）区域质心的方向为向量的方
#向。为了进一步提高旋转不变性，要计算以角点为中心半径为 r 的圆形区域的矩，再根据矩计
#算出方向.

#对于描述符，ORB 使用的是 BRIEF 描述符。但是我们已经知道 BRIEF对与旋转是不稳定的。所以
#我们在生成特征前，要把关键点领域的这个 patch的坐标轴旋转到关键点的方向。在描述符匹配中
#使用了对传统 LSH 改善后的多探针 LSH。
'''
#OpenCv中的ORB算法
img = cv2.imread('buildingblocks.jpg',0)
#cv2.ORB()有几个参数可选,
#最有用的应该是nfeature,默认值为500,它表示了要保留特征的最大
#scoreType设置使用Harris还是FAST打分对特征进行排序(默认Harris)
#参数WTA_K决定产生每个oriented_BRIEF描述符要使用的像素点数目,默认值为2,一次使用两个点
#在这种情况下匹配,要使用NORM_HAMMING 距离。如果 WTA_K 被设置成 3 或 4，那匹配距离就要
#设置为 NORM_HAMMING2。
orb = cv2.ORB_create()

#注意以上代码还有一种写法
"""
orb = cv2.ORB()
orb = orb.create()
"""

kp = orb.detect(img,None)

kp,des = orb.compute(img,kp)
#设置输出图像为None
img2 = cv2.drawKeypoints(img,kp,None)
plt.imshow(img2),plt.show()
'''

############
# 特征匹配 #
############

#Brute-Force匹配的基础
##蛮力匹配器是很简单的。首先在第一幅图像中选取一个关键点然后依次与第二幅图像的每个关键
##点进行（描述符）距离测试，最后返回距离最近的关键点。


#对于BF匹配器,首先使用cv2.BFMatcher()创建一个BF-Matcher对象.它有两个可选参数,第一个是
#normType,它是用来指定要使用的距离测试类型,默认值为cv2.Norm_L2. 这很适合SIFT和SURF等
#(cv2.NORM_L1也可以). 对于使用二进制描述符的ORB, BRIEF, BRISK算法等,要使用cv2.NORM_HAMMING.
#这样就会返回两个测试对象之间的汉明距离. 如果ORB算法的参数设置为VTA_K==3或4，normType
#就应该设置成cv2.NORM_HAMMING2
#第二个参数为布尔变量crossCheck,默认值为False,如果设置为True,匹配条件就会更加严格,只有
#到A中的第i个特征点与B中的第j个特征点距离最近,并且B中的第j个特征点到A中的第i个特征点也
#最近(A中没有其他店到j的距离更近)时才会返回最佳匹配(i,j). 也就是这两个特征点要相互匹配
#才行. 这样就能提供统一的结果,这可以用来替代D.Love在SIFT文章中提出的比值测试方法.

#BFMatcher对象具有两个方法,BFMatcher.match()和BFMatcher.knnMatch()
#第一个方法返回最佳匹配,第二个方法为每个关键点返回k个最佳匹配(降序排列之后取前k个).其中
#k是由用户设定的,如果出了匹配之外还要做其他事情的话可能会用上(比如进行比值测试)
#就像使用cv.drawKeypoints()绘制关键点一样,我们可以使用cv2.drawMatches()来绘制匹配的点.
#它会将这两幅图像先水平排列,然后在最佳匹配的点之间绘制直线(从原图像到目标图像),如果前面
#使用的是BFMatcher.knnMatch(). 现在我们可以使用函数cv2.drawMatchesKnn()为每个关键点和它
#的k个最佳匹配点绘制匹配线.如果k等于2，就会为每个关键点绘制两条最佳匹配直线, 如果我们要
#选择性绘制就要给函数传入一个掩膜.

#img1 = cv2.imread('aroundCity.jpg',0)
#img2 = cv2.imread('aroundCity_scene.jpg',0)

#对ORB描述符进行蛮力匹配
'''
orb = cv2.ORB_create()
#计算两幅图像的关键点与ORB描述符
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)
#创建一个BFMatcher对象,并将距离设置为cv2.NORM_HAMMING(因为这里使用的是ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#使用match()获取两幅图像的最佳匹配,返回的是一个DMatch对象列表,这个DMatch对象具有以下属性
#DMatch.distance：描述符之间的距离,越小越好
#DMatch.trainIdx：目标图像中描述符的索引
#DMatch.queryIdx：查询图像中描述符的索引
#DMatch.imgIdx：目标图像的索引
matches = bf.match(des1,des2)
#将匹配结果按特征点之间的距离进行降序排列,使最佳匹配排在前面
matches = sorted(matches, key = lambda x: x.distance)

img_result = cv2.drawMatches(img1,kp1,img2,kp2,matches[0:10],None,flags=2)

plt.imshow(img_result,'gray'),plt.show()
cv2.imwrite('BF_ORB_RESULT.jpg',img_result)
'''
#对SIFT描述符进行蛮力匹配和比值测试
#注意,涉及到SIFT,此处已经解决opencv-contrib-python的模块问题,可以正常使用
'''
sift = cv2.xfeatures2d.SIFT_create()
#特征检测器改为SIFT
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
#使用knnMatch来获得k对最佳匹配,这里设置k=2
matches = bf.knnMatch(des1,des2,k=2)
#比值测试,首先获取与A距离最近的点B(最近)和C(次近),只有当B/Cx小于阈值时(0.75)才被认为是
#匹配的,因为假设匹配时一一对应的,真正的匹配的理想距离为0
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append((m,n))
#注意这里的drawMatchesKnn中matches(第五个参数)的形式应该是一个二维列表,每个索引都是k个值        
img_knn_result = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)

cv2.imwrite('BF_SIFT_KnnResult.jpg',img_knn_result)
'''
#FLANN匹配器:快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbour),它是一个
#############对大数据集和高维特征进行最近邻搜索的算法的集合,而且这些算法都已经被优化过
#############了. 在面对大数据集时它的效果要好于BFMatcher。

#使用FLANN匹配,我们需要传入两个字典作为参数. 这两个用来确定要使用的算法和其他相关参数等
#第一个是IndexParams,各种不同的算法的信息可以在FLANN文档中查到. 这里我们总结一下,对于
##SIFT和SURF等,我们可以传入的参数是:dict(alogorithm = FLANN_INDEX_KDTREE,trees=5),但使
##用ORB时,我们要传入的参数为?
#第二个字典是SearchParams. 用它来制定递归遍历的次数,值越高结果越准确,但是消耗的时间也越
##多,如果你想修改这个值,传入参数:searchparams = dict(checks=100)
'''
sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)
#设置FLANN的两个参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
#只需要绘制较好的匹配结果,所以这里建立一个掩膜
matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
#使用一个字典将绘制参数打包
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),flags = 0,\
                   matchesMask = matchesMask)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,'gray')
plt.show()
'''

################################ 上一节我们在杂乱的图像中找到了一个对象(的某些部分)的位
# 使用特征匹配和单应性查找对象 # 置,这些信息足以帮助我们在目标图像中准确的找到(查询图像)
################################ 对象

#为了达到这个目的我们可以使用calib3d模块中的cv2.findHomography()函数. 如果将这两幅图像
#中的特征点集传给这个函数,它就会找到这个对象的透视图变换. 然后我们就可以使用函数
#cv2.perspectiveTransform()找到这个对象了. 至少要4个正确的点才能找到这种变换.

#我们已经知道在匹配过程中可能会有一些错误,而这些错误会影响最终结果,为了解决这个,算法使用
#RANSAC和LEAST_MEDIAN（可以通过参数设定). 所以好的匹配提供的正确的估计被称为inliers,剩
#下的被称为outliers. cv2.findHomography()返回一个掩膜,这个掩膜确定了inlier和outlier点
'''
img1 = cv2.imread('aroundCity.jpg',0)
img2 = cv2.imread('aroundCity_scene.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
#现在我们呢设置只有存在10个以上匹配时才去查找目标(MIN_MATCH_COUNT),否则显示警告消息
#"现在匹配不足"
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    #如果找到了足够的匹配,我们要提取两幅图像中匹配点的坐标,把它们传入到函数中计算透视
    #变换. 一旦我们找到 3x3 的变换矩阵,就可以使用它将查询图像的四个顶点(四个角)变换到
    #目标图像中去了,然后再绘制出来

    #m.queryIdx查询对象中描述符的索引,kp1[m.queryIdx].pt获取点的坐标
    #reshape(),第一个参数为重建后的数组长度,-1代表根据剩余元素自动计算
    #第二,三个元素为shape,这里是将每个值重构为 1x2的形式
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    #m.trainIdx目标图像中描述符的索引,同上操作后得到一个目标对象中匹配点的坐标列表
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    #第三个参数是计算单应性矩阵的方法,可选cv2.RANSAC,cv2.LMEDS
    #第四个参数取值范围在1到10, 拒绝一个点对的阈值,原图像的点经过变换后点与目标图像上
    #对应点的误差. 超过这个误差就认为是outlier
    #返回的M为单应性矩阵, mask掩膜确定outlier和inlier点
    M,mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    #将查询图像的四个点(四个角)变换到目标图像中去
    h,w = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    #根据变换后四个角点绘制查询对象在目标对象中的一个大致矩形
    cv2.polylines(img2,[np.int32(dst)],True,255,10,cv2.LINE_AA)
else:
    #没有找到足够的匹配点
    print("Not enough matches are found -{}/{}".format(len(good),MIN_MATCH_COUNT))
    matchesMask = None
#绘制inliers或者匹配关键点(如果匹配失败）
draw_params = dict(matchColor=(0,255,0),singlePointColor=None,matchesMask=matchesMask\
                   ,flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3,'gray'),plt.show()
'''

############
# 视频分析 #
############

#######################
# Meanshift和Camshift #
#######################

#Meanshift：Meanshift算法的基本原理是很简单的. 假设我们有一堆点(比如直方图反向投影得到
############的点),和一个小的圆形窗口,我们要完成的任务就是将这个窗口移动到最大灰度密度处
############(或者是点最多的地方). 按照这样的操作我们的窗口最终会落在像素值(和)最大的地方
############通常情况下我们要使用直方图反向投影得到的图像和目标对象的起始位置. 当目标对
############象的移动会反映到直方图反向投影图中. 就这样,meanshift算法就把我们的窗口移动
############到图像中灰度密度最大的区域了.

#OpenCv中的Meanshift
#要在Opencv中使用Meanshift算法首先我们要对目标对象进行设置,计算目标对象的直方图,这样在
#执行meanshift算法时我们就可以将目标对象反向投影到每一帧中去了. 另外我们还需要提供窗口
#的起始位置. 在这里我们值计算H(hue)通道的直方图,同样为了避免低亮度造成的影响,我们使用
#函数cv2.inRange()将低亮度的值忽略掉.
def my_resize(img,times):
    h,w = img.shape[:2]
    img = cv2.resize(img,(int(w*times),int(h*times)),interpolation = cv2.INTER_CUBIC)
    return img

def draw_rectangle(event,x,y,flags,param):
    global x0,y0,x1,y1,draw
    if event == cv2.EVENT_LBUTTONDOWN and draw == False:
        x0,y0 = x,y
        draw = True
    elif event == cv2.EVENT_LBUTTONUP and draw == True:
        x1,y1 = x,y
        cv2.rectangle(frame1,(x0,y0),(x1,y1),(0,255,0),2)
        draw = False
        
cap = cv2.VideoCapture('London_car_stream.mov')

ret,frame = cap.read()
#划定初始矩形框位置
x0,y0,x1,y1 = 0,0,0,0
draw = False
times = 0.5
cv2.namedWindow('car_stream')
frame1 = my_resize(frame,times)
cv2.setMouseCallback('car_stream',draw_rectangle)
while(1):
    cv2.imshow('car_stream',frame1)
    if cv2.waitKey(0) == 27:
        break
#x0,y0,x1,y1 = int(x0/times),int(y0/times),int(x1/times),int(y1/times)
track_window = (x0,y0,abs(x1-x0),abs(y1-y0))
#设置roi窗口用于追踪
roi = frame1[y0:y1,x0:x1]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#将饱和度60以下,明度32以下的点忽略
mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#设置迭代终止标准,直到窗口中心移位小于标准或迭代十次,每次至少移动一个像素
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
i = 0
while(1):
    ret,frame = cap.read()
    frame = my_resize(frame,0.5)
    if ret == True :
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #做直方图的反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,100],1)
        #cv2.meanShift(),返回迭代次数和迭代后的窗口
        #第一个参数为对象直方图的反向投影
        #第二个参数为初始搜索窗口
        #第三个参数为停止迭代的标准
        ret,track_window = cv2.meanShift(dst,track_window,term_crit)

        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x+x1-x0,y+y0-y1),255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(0)
        if k == 27:
            break
        else:
            i = i+1
            #cv2.imwrite('meanShift_camShift/car_{}.jpg'.format(i),img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()

#Camshift
#上面的结果存在一个问题,我们的窗口大小是固定的,而汽车由远及近(在视觉上)是一个逐渐变大的
#过程,固定的窗口是不合适的,所以我们需要根据目标的大小和角度来对窗口的大小和角度进行修正
#Camshift算法为我们带来了解决方案. 这个算法首先要使用meanshift,meanshift找到(并覆盖目标
#之后,再去调整窗口大小. 它还会计算目标对象的最佳外接椭圆的角度,并以此调节窗口角度. 然后
#使用更新后的窗口大小和角度来在原来的位置继续进行meanshift,重复这个过程指导达到需要的精度

#Opencv中的Camshift
#与meanshift基本一样,但是返回的结果是一个带有旋转角度的矩形,以及这个矩形的参数(被用到下
#一此迭代过程中)

'''
cap = cv2.VideoCapture('London_car_stream.mov')

ret,frame = cap.read()

#setup initial location of window
x0,y0,x1,y1 = 0,0,0,0
draw = False
times = 0.5
cv2.namedWindow('car_stream')
frame1 = my_resize(frame,times)
cv2.setMouseCallback('car_stream',draw_rectangle)
while(1):
    cv2.imshow('car_stream',frame1)
    if cv2.waitKey(0) == 27:
        break
x0,y0,x1,y1 = int(x0/times),int(y0/times),int(x1/times),int(y1/times)
track_window = (x0,y0,x1-x0,y1-y0)
#设置roi窗口用于追踪
roi = frame[y0:y1,x0:x1]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#将饱和度60以下,明度32以下的点忽略
mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#设置迭代终止标准,直到窗口中心移位小于标准或迭代十次,每次至少移动一个像素
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
i = 0
while(1):
    ret,frame = cap.read()

    if ret == True :
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #做直方图的反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,100],1)
        #cv2.CamShift(),返回一个带有旋转角度的矩形和这个矩形的参数
        ret,track_window = cv2.CamShift(dst,track_window,term_crit)
        #cv2.boxPoints()接受的参数为一个旋转的矩形,输出矩形的四个顶点
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        #以四个顶点绘制变化后的矩形
        img2 = cv2.polylines(frame,[pts],True,(0,255,0),2)
        img2 = my_resize(img2,times)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(0)
        if k == 27:
            break
        else:
            i = i+1
            cv2.imwrite('meanShift_camShift/car_{}.jpg'.format(i),img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()
'''


