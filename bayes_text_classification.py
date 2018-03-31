
# coding: utf-8

# In[81]:


import numpy as np
import math


# In[82]:


#【1】准备数据：编写3个函数：
#1）第一个函数将留言文本切分成一系列词条集合，去除标点；也返回类别标签集合
#2）第二个函数创建包含所有文档中出现的不重复词的列表
#3）第三个函数，通过词汇表来把某个新的文档，输出为文档向量


# In[83]:


#【补充知识点】：
#参考：Iteration to make a union of sets ：https://stackoverflow.com/questions/37355381/iteration-to-make-a-union-of-sets
#参考：set与list搭配使用：https://blog.csdn.net/zongzhiyuan/article/details/50099657
#参考：列表list,元组Tuple,字典Dict,集合Set：https://blog.csdn.net/liuyanfeier/article/details/53731239
#参考：Python set won't perform union：https://stackoverflow.com/questions/19580944/python-set-union-and-set-intersection-operate-differently
#参考：What does |= (ior) do in Python?：https://stackoverflow.com/questions/3929278/what-does-ior-do-in-python
#参考：python_list用法：http://www.runoob.com/python/att-list-index.html
#参考：修改list值出现错误，参考：https://www.cnblogs.com/jiangzhaowei/p/5740913.html
#参考：numpy.log: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.log.html


# In[84]:


# 1)定义loadDataSet()创建一些实验样本
def createDataSet():
    #postingList是进行词条切分后的文档合集，文档来自斑点犬留言板，这些留言文本被切分为词条合集
    #假设数据为最简单的6篇文章，每篇文章大概7~8个词汇左右，如下
    postingList = [['my','dog','has','flea','problem','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','i','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 2)统计每篇文章中的出现的单词列表，列表词条不重复，取并集汇总
def createVocabList(dataSet):
    vocab = set([]) 
    for document in dataSet:
        vocab |= set(document)
    return list(vocab)

# 3）获得词汇表后，使用setOfWord2Vec()函数,输入参数为词汇表及某篇文章，输出该篇文档向量，向量中每个元素为1或者0，1为侮辱性文章
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        #参考python3.0中打印的用法：https://stackoverflow.com/questions/14753844/python-3-using-s-and-format
        #else: print("The word %s is not in my Vocabulary" &word) python3 不再支持这种写法
        else: print("The word {} is not in my Vocabulary".format(word))
    return returnVec


# In[85]:


#试着运行一下上方的函数
data1, class1 = createDataSet() #执行第一步,让函数1返回的两个值赋给 data1, 与 class1
myVecabList = createVocabList(data1) # 执行第二步，调用函数2统计出文档中出现的不重复词汇的列表


# In[86]:


data1[0]


# In[87]:


myVecabList


# In[88]:


len(myVecabList)


# In[89]:


returnVec1 = setOfWords2Vec(myVecabList, data1[0])


# In[90]:


#首先取vacalist中的第一个值“mr”，把它与input文章中每一个单词对比，如果没有任何一个相同，运行一次print，不影响函数3中return结果
#取第二个值“help”，拿着它与input新文章中的每一个单词对比，发现有相同的，返回index(word)
returnVec1


# #朴素贝叶斯分类器训练函数

# In[99]:


#【2】训练算法：朴素贝叶斯分类器训练函数
from math import log
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #获得训练集中文章个数
    numWords = len(trainMatrix[0]) #获得训练集中单词数量，这里序号0是指第一篇文章的单词数量
    pAbusive = sum(trainCategory)/float(numTrainDocs) #计算p(Ci),abusive侮辱的意思，计算侮辱文档所占比例
    p0Num = np.ones(numWords) #初始化概率的分子变量， 这里修改了，
    p1Num = np.ones(numWords) 
    #numpy.zeros(),参考博客:https://blog.csdn.net/qq_26948675/article/details/54318917
    #用法：zeros(shape, dtype=float, order='C'); 返回：返回来一个给定形状和类型的用0填充的数组；
    p0Denom = 2.0 #初始化值，概率的分母变量
    p1Denom = 2.0
    
    for i in range(numTrainDocs): #遍历每一篇文章
        if trainCategory[i] == 1: #先判断该篇文章是否被标记为侮辱性文章
            p1Num += trainMatrix[i] #侮辱词汇计数加一
            p1Denom += sum(trainMatrix[i]) #文档总词数加一
        else: #如果该篇文章非侮辱性
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #计算每个词出现侮辱性词汇的概率 p(W_i|c_1), 注意这里修改为了np.log:太多太小的数相乘，程序会下溢，取对数可以避免
    p0Vect = np.log(p0Num/p0Denom) #计算每个词出现非侮辱性词汇的概率 p(w_i|c_0)
    return p0Vect, p1Vect, pAbusive
        


# In[106]:


#试着运行一下上方的函数
data1, class1 = createDataSet() #执行第一步,让函数1返回的两个值赋给 data1, 与 class1
myVecabList = createVocabList(data1) # 执行第二步，调用函数2统计出文档中出现的不重复词汇的列表
trainMat = [] #利用for循环填充trainMat列表，每一行对应的词向量
for postinDoc in data1:
    trainMat.append(setOfWords2Vec(myVecabList, postinDoc))
    
p0V,p1V,pAb = trainNB0(trainMat,class1)


# In[107]:


#看下一值
pAb #侮辱性文章的概率为0.5，因为数据刚开始已经标注好了，3篇为侮辱性，3篇为非侮辱性


# In[108]:


p0V #p(w_i|c_1)


# In[109]:


p1V #p(w_i|c_0)


# In[110]:


#【测试算法】
#利用贝叶斯分类器对文档进行分类时，要计算过个概率的乘积以获得文档属于某个类别的概率，如果其中一个概率值为0，那么最后的乘积也为0。为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。所以需要修改一下trainNB()中的分母分子初始化代码。


# In[105]:


#初始化概率的分子变量
#p0Num = ones(numWords); p1Num = ones(numWords)
#初始化概率的分母变量
#p0Denom = 2.0; p1Denom = 2.0


# In[115]:


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #vec2Classify：文档矩阵，p0Vec：非侮辱性词汇概率向量p1Vec：侮辱性词汇概率向量
    #向量元素相乘后求和再加到类别的对数概率上，等价于概率相乘
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#朴素贝叶斯分类器测试函数
def testingNB():
    #调入数据
    data1, class1 = createDataSet()
    #构建一个包含所有词的列表
    myvocabList = createVocabList(data1)
    #初始化训练数据列表
    trainMat = []
    #填充训练数据列表
    for postinDoc in data1:
        trainMat.append(setOfWords2Vec(myVecabList, postinDoc))
    #训练
    p0V,p1V,pAb = trainNB0(trainMat,class1)
    #测试
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    #测试
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# In[116]:


testingNB()

#准备数据：文档词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋到向量的转换

    vocabList：词袋
    inputSet：某个文档

    returnVec：文档向量
    """
    #创建一个所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    #将新词集合添加到创建的集合中
    for word in inputSet:
        #如果文档中的单词在词汇表中，则相应向量位置加1
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    #返回一个包含所有文档中出现的词的列表
    return returnVec
