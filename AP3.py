# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:48:27 2020

@author: trist
"""


from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 生成测试数据
centers = [[1, 1], [-1, -1], [1, -1]]
# 生成实际中心为centers的测试样本300个，X是包含300个(x,y)点的二维数组，labels_true为其对应的真实类别标签
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)    
##############################################################################
#X = np.loadtxt(open("C:/Users/trist/Desktop/AP/data33b.csv","rb"),delimiter=",")  
#labels_true = np.loadtxt(open("C:/Users/trist/Desktop/AP/yb.csv","rb"),delimiter=",")  
 
##############################################################################
 
 
def euclideanDistance(X, Y):
	"""计算每个点与其他所有点之间的欧几里德距离"""
	X = np.array(X)
	Y = np.array(Y)
	# print X
	return np.sqrt(np.sum((X - Y) ** 2))
 
 
 
def computeSimilarity(datalist):
 
	num = len(datalist)
 
	Similarity = []
	for pointX in datalist:
		dists = []
		for pointY in datalist:
			dist = euclideanDistance(pointX, pointY)
			if dist == 0:
				dist = 1.5
			dists.append(dist * -1)
		Similarity.append(dists)
 
	return Similarity
 
 
def affinityPropagation(Similarity, lamda):
 
	#初始化 吸引矩阵 和 归属 矩阵
	Responsibility = np.zeros_like(Similarity, dtype=np.int)
	Availability = np.zeros_like(Similarity, dtype=np.int)
 
	num = len(Responsibility)
 
	count = 0
	while count < 10:
		count += 1
		# update 吸引矩阵
		for Index in range(num):
			# print len(Similarity[Index])
			kSum = [s + a  for s, a in zip(Similarity[Index], Availability[Index])]
			# print kSum
			for Kendex in range(num):
				kfit = delete(kSum, Kendex)
				# print fit
				ResponsibilityNew = Similarity[Index][Kendex] - max(kfit)
				Responsibility[Index][Kendex] = lamda * Responsibility[Index][Kendex] + (1 - lamda) * ResponsibilityNew
		# print "Responsibility", Responsibility
		# update 归属矩阵
		ResponsibilityT = Responsibility.T
		# print ResponsibilityT, Responsibility
		for Index in range(num):
			iSum = [r for r in ResponsibilityT[Index]]
			for Kendex in range(num):
				# print Kendex
				# print "ddddddddddddddddddddddddddd", ResponsibilityT[Kendex]
				#
				ifit = delete(iSum, Kendex)
				ifit = filter(isNonNegative, ifit)   #上面 iSum  已经全部大于0  会导致  delete 下标错误
				#   k == K 对角线的情况
				if Kendex == Index:
					AvailabilityNew  = sum(ifit)
				else:
					result = Responsibility[Kendex][Kendex] + sum(ifit)
					AvailabilityNew = result if result > 0 else 0
				Availability[Kendex][Index] = lamda * Availability[Kendex][Index] + (1 - lamda) * AvailabilityNew
		print ("###############################################")
		print (Responsibility)
		print (Availability)
		print ("###############################################")
	return Responsibility + Availability
 
def computeCluster(fitable, data):
	clusters = {}
	num = len(fitable)
	for node in range(num):
		fit = list(fitable[node])
		key = fit.index(max(fit))
#       if not clusters.has_key(key): 
		if not key in clusters:        
			clusters[key] = []
		point = tuple(data[node])
		clusters[key].append(point)
 
	return clusters
##############################################################################
 
"""切片删除 返回新数组"""
def delete(lt, index):
	lt = lt[:index] + lt[index+1:]
	return lt
 
def isNonNegative(x):
	return x >= 0
 
 
##############################################################################
 
Similarity = computeSimilarity(X)
 
Similarity = np.array(Similarity)
 
print ("Similarity", Similarity)
 
fitable = affinityPropagation(Similarity, 0.34)
 
print (fitable)
 
clusters = computeCluster(fitable, X)
 
# print clusters
 
##############################################################################
clusters = clusters.values()
 
print (len(clusters))
 
##############################################################################
def plotClusters(clusters, title):
	""" 画图 """
	plt.figure(figsize=(8, 5), dpi=80)
	axes = plt.subplot(111)
	col=[]
	r = lambda: random.randint(0,255)
	for index in range(len(clusters)):
		col.append(('#%02X%02X%02X' % (r(),r(),r())))
	color = 0
	for cluster in clusters:
		cluster = np.array(cluster).T
		axes.scatter(cluster[0],cluster[1], s=20, c = col[color])
		color += 1
	plt.title(title)
	# plt.show()
##############################################################################
plotClusters(clusters, "clusters by affinity propagation")
plt.show()
 
##############################################################################
