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
print(experiment_data.shape)
plt.scatter(experiment_data[:,0],experiment_data[:,1])
plt.show()

x=7*np.random.rand()
y=2.5*np.random.rand()
print(x,y)
xx,yy=experiment_data.shape
print(xx,yy)
def kmeans(data,k=2):

    #  生成简陋的随机数
    def _rand_center(data,k):
        x1 = 7 * np.random.rand()
        y1 = 2.5 * np.random.rand()
        x2 = 7 * np.random.rand()
        y2 = 2.5 * np.random.rand()
        sca=np.zeros((2,2))
        sca[0,0]=x1
        sca[0,1]=y1
        sca[1,0]=x2
        sca[1,1]=y2
        return sca
    def _distance(p1,p2):
        tmp=np.sum((p1-p2)**2)
        return np.sqrt(tmp)
    def _end_process(sca1,sca2):
        set1=set([tuple(c) for c in sca1])
        set2=set([tuple(c) for c in sca2])
        return (set1 == set2)

    data=experiment_data
    sca_ini=_rand_center(data,2)
    xx,yy=data.shape
    label=np.zeros(xx,1)
    process=False
    new_sca=sca_ini
    k=2
    while not process:
        old_sca =np.copy(new_sca)
        min_dist,min_index=np.inf,-1
        for i in range(xx):
            for j in range(k):
                dist=_distance(data[i],new_sca[j])
                if dist<min_dist:
                    min_dist,min_index=dist,j
                    label[i]=j


# 在这里随机点难道一直随机生成么？
#  今晚好像暂时做不出来了
