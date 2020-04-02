# 这是基础K-means算法的验证
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#  下面构建鹫尾花数据集的分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 在这里使用鹫尾花数据集对基础k-means算法进行验证

# 导入数据集
iris_dataset=load_iris()
print(iris_dataset.keys())

X, X_test, y, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# y轴的数据可以进行忽略操作

# 首先选取维度2 3 进行绘图  之后进行分类即可
experiment_data=X[:,2:4]
'''print(experiment_data.shape)'''
xx=experiment_data[:,0]
yy=experiment_data[:,1]

#初始化点
def get_points(xx,yy,k):
    maxx=np.max(xx)
    mix=np.min(xx)
    may=np.max(yy)
    miy=np.min(yy)
    points=np.zeros((k,2))
    for i in range(k):
        x=np.random.rand()*(maxx-mix)+mix
        y=np.random.rand()*(may-miy)+miy
        points[i,0]=x
        points[i,1]=y

    return points

tmp=get_points(xx,yy,2)
k=2
for i in range(k):
    plt.scatter(tmp[i,0],tmp[i,1],color='red')

plt.scatter(experiment_data[:,0],experiment_data[:,1],color='blue')
plt.show()

#求出两点间的距离
def distance(p1,p2):
    det_x=p1[0]-p2[0]
    det_y=p1[1]-p2[1]
    tmp = (det_x** 2)+(det_y**2)
    return np.sqrt(tmp)
#下面写分类函数
def type(cen1,cen2,p):
    dis1=distance(p,cen1)
    dis2=distance(p,cen2)
    if dis1>dis2:
        return 1
    else:
        return 2

def total_type(experiment_data,tmp):
    xx,yy=experiment_data.shape
    ttyypp=np.zeros((xx,1))
    for i in range(xx):
        ttyypp[i,0]=type(tmp[0,:],tmp[1,:],experiment_data[i,:])
    return ttyypp
    #以上操作完成分类

def renew(ttyypp,data,k):
    xx,yy=data.shape
    x1=y1=x2=y2=0
    points = np.zeros((k, 2))
    c1=c2=0;
    for i in range(xx):
        if ttyypp[i,0]==1:
            x1+=data[i,0]
            y1+=data[i,1]
            c1+=1
        else:
            x2+=data[i,0]
            y2+=data[i,1]
            c2+=1
    points[0,0]=x1/c1
    points[0,1]=y1/c1
    points[1,0]=x2/c2
    points[1,1]=y2/c2
    return points


#下面是主循环
xx=experiment_data[:,0]
yy=experiment_data[:,1]
tmp=get_points(xx,yy,2)
k=2
ttyypp=np.zeros((112,1))
for i in range(10000):
    ttyypp=total_type(experiment_data,tmp)
    tmp=renew(ttyypp,experiment_data,k)



for i in range(2):
        plt.scatter(tmp[i,0],tmp[i,1],color='red',s=100)
for i in range(112):
    if ttyypp[i]==1:
        plt.scatter(experiment_data[i:,0],experiment_data[i:,1],color='g')
    else:
        plt.scatter(experiment_data[i:,0],experiment_data[i:,1],color='b')

plt.show()