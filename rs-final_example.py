# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 为了实现一个RS
# ## 1.计算相似性
# ## 2.选择邻居
# ## 3.预测
# %% [markdown]
# #
# %% [markdown]
# # 为了实现一个RS
# ## 1.数据导入(csv->pandas)
# ## 2.数据存储(分组：直接切开10份可以吗？训练集和测试集如何分？)
# ## 3.数据检索（为什么要检索数据，检索什么数据）
# ## 4.中间结果储存（dict）
# ## 5.计算相似性
# ## 6.选择邻居
# ## 7.预测评分
# ## 8.度量
# ## 9.推荐列表
# 
# %% [markdown]
# 1. 读入数据(F)
# 2. 切块分组(I)
# 3. 确定邻居数量(G)
# 4. 用户配对(D)
# 5. 用户，物品，评分索引(E)
# 6. 获取某用户评分过物品的函数(B)
# 7. 获取user item，的评分(A)
# 8. 计算皮尔逊相似性函数(C)
# 9. 按照配对计算相似性，保存起来(E)
# 10. 邻居排序(J)
# 11. 定义邻居数据单元(I)
# 12. 预测公式(L)
# 13. 计算所有用户平均评分(A)
# 14. 针对u，i对，进行评分预测(L)
# 15. 计算误差(N)
# %% [markdown]
# 预测需要--> 平均分(1)，相似性(2)，邻居集(3)
# 验证预测，需要-->需要分训练集(4)和测试集(5)
# ------------
# 预测需要预测公式(L)
# 验证预测需要误差公式(N)
# 平均分需要(1)，可以获取给定u的所有评分(A)
# 相似性需要(2)，给定user，获得他的所有评分物品(B)，和对这些物品的评分(A)
# 相似性需要(2)，计算皮尔逊相似性函数(C)
# 计算所有用户两两相似性(2)，需要指导有多少用户对pairs(D),需要保存相似性(E)
# 获取评分和物品集合(A,B)，需要 把评分存起来，还能快速获取(F)
# 储存数据，需要读取数据(G)
# -----
# 邻居集需要相似性排序(3)，需要k做参数(H)，需要一个定义数据集(I),需要相似性排序(J)
# 分训练集(4)和测试集(5),需要随机分组(K)
# 

# %%
#1. 读入数据(F)
#导入pandas库
import pandas as pd
ratings=pd.read_csv("ratings.csv")
#print(ratings)

items=pd.read_csv("movies.csv",sep=',')
#print(items.head(5))

users=set(ratings['userId'])
#print(users)


# %%
#测试
u=ratings.loc[90096]
u.loc['userId']
print(u,'\n\n',int(u['userId']),int(u['movieId']),u['rating'])
print(u,'\n\n',int(u[0]),int(u[1]),u[2])


# %%
#2. 切块分组(I)
import random
import math
rating_index_set=[(i[1],i[2]) for i in (ratings.loc[:,['userId','movieId']].itertuples())]
print(max(rating_index_set))
random.shuffle(rating_index_set)
#print(rating_index_set)
#----------------------------
#一共有m块，自动分（尽可能平均）
#split the arr into N chunks
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]
folds=chunks(rating_index_set,10)
for i in folds:
    print(len(i),i[0:4])
#----------------------------
folds=folds

#3. 确定邻居数量(G)
k = 40 # number of neighborsy    

# %% [markdown]
# itertools.combination([1,2,3])

# %%
#4. 用户配对(D)
import itertools
pairs=[i for i in itertools.combinations(users,2)]
for i in pairs:
    print(i)

# %% [markdown]
# # 5. 用户，物品，评分索引(E)
# # let us make index

# %%
# user -> item -> rating
dict_uir={}
#dict_iur={}
for r in ratings.itertuples():
    index_ = r.Index
    user=int(r.userId)
    item=int(r.movieId)
    rating=r.rating
    if user in dict_uir:
        d=dict_uir[user]
        d[item]=rating     
    else:
        d={item:rating}
        dict_uir[user]=d    
# 测试
print(dict_uir[1])


# %%
# 6. 获取某用户评分过物品的函数(B)
def getItemsBy (dict_uir:dict,user:int):
    
    if user in dict_uir:
        return set(dict_uir[user].keys())
    else:
        None


# %%
# 测试
getItemsBy(dict_uir,1)


# %%
# 7. 获取user item，的评分(A)
def getRating(dict_uir:dict,user:int,item:int):
    if user in dict_uir:
        if item in dict_uir[user]:
            return dict_uir[user][item]
        else :
            None
    else:
        None


# %%
# 测试
getRating(dict_uir,1,1)

# %% [markdown]
# # similarity calculation

# %%
#8. 计算皮尔逊相似性函数(C)
#import numpy as np
import math
def cal_pccs(x, y):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    n1=len(x)
    n2=len(y)
    if n1!=n2 or n1 == 0 or n2 == 0:
        return None
    else:
        ave_x=   sum(x) / len (x) # np.average(x)
        ave_y=  sum(y) / len (y) # np.average(y)
        x_num = [i - ave_x  for i in x]
        y_num = [i - ave_y  for i in y]
        num=[x*y for (x,y) in list(zip(x_num,y_num))]
        num_sum= sum(num)
        x_den = [i * i  for i in x_num]
        y_den = [i * i  for i in y_num]
        den=math.sqrt(sum(x_den)*sum(y_den))           
        if den==0.0:
            return None 
        else :
            return num_sum/den


# %%
# 保存相似性
# 9. 按照配对计算相似性，保存起来(E)

similarity={}
for (u,v) in pairs:
    #print(u,v)    
    items_by_u = getItemsBy(dict_uir,u)
    items_by_v = getItemsBy(dict_uir,v)
    #print(len(items_by_u),len(items_by_v))
    if len(items_by_u) >0 and len(items_by_v) >0:        
        intersected = items_by_u.intersection(items_by_v)
        #print(intersected)
        if len(intersected)>0:
            ratings_u = [getRating(dict_uir,u,i) for i in intersected]
            ratings_v = [getRating(dict_uir,v,i) for i in intersected]
            #print(len(ratings_u),len(ratings_v))
            if ratings_u !=None and ratings_v !=None:                 
                #s=np.corrcoef(ratings_u,ratings_u)                   
                s = cal_pccs(ratings_u,ratings_v)
                if s==None or math.isnan(s):
                    ()
                else:
                    #print(s)
                    if u in similarity:
                        similarity[u][v]=s
                    else :
                        d={v:s}  
                        similarity[u]=d
                    if v in similarity:
                        similarity[v][u]=s
                    else :
                        d={u:s}  
                        similarity[v]=d  
                
            else: () 
        else:
            ()
    else:
        ()
print(similarity[1])



# %%
# from multiprocessing import Pool
# pool = Pool()
# def Test(a,b):
#     return(a,b,a+b)

# data=[(1,2),(3,4),(1,2),(3,4),(1,2),(3,4),(1,2),(3,4),(1,2),(3,4),(1,2),(3,4),(1,2),(3,4),(1,2),(3,4)]
# pool.map(Test,data)


# %%
# 10. 邻居排序(J)

# neighbor selection
neighbors = {}
for s in similarity:
    neigh=similarity[s]
    r=[(k,neigh[k]) for k in neigh]
    r=sorted(r,key=lambda i:i[1],reverse=True)
    #print(r)
    neighbors[s] = r
print(neighbors[1])


# %%
# 11. 定义邻居数据单元(I)

class NeighborInfo():    
    def __init__(self,neighbor_id,rating_on_target,similarity):
        self.Neighbor_id=neighbor_id
        self.Rating=rating_on_target
        self.Similarity=similarity
        
    


# %%

# 12 预测公式
def predict(user_average_rating_dict:dict,target_user:int,neighbors:list):
    ave_u  = 3.5
    if target_user in user_average_rating_dict:
        ave_u = user_average_rating_dict[target_user]                
    numerator=0.0
    denominator =0.0
    for n in neighbors:
        ave_v=3.5
        if n.Neighbor_id in user_average_rating_dict:
            ave_v=user_average_rating_dict[n.Neighbor_id] 
        numerator = numerator + (n.Rating - ave_v) * n.Similarity
        denominator = denominator + math.fabs(n.Similarity)
    r = 0.0 
    if denominator!=0.0:
        r=numerator/denominator
    if math.isnan(r) :
        return ave_u
    else:
        return (ave_u + r)


# %%
#12. 计算所有用户平均评分(A)

ave_rating={}
for u in dict_uir:
    ir=dict_uir[u].values()
    ave=sum(ir )/len(ir)
    ave_rating[u]=ave
print(ave_rating[1])


# %%
#13. 针对u，i对，进行评分预测(L)

prediction={}
for fold in folds:
   testing_set= set(fold)
   for (user,item) in fold:
      result=[]      
      if user in neighbors:
        for (user2,sim) in neighbors[user]:
            if (user2 in dict_uir) and item in dict_uir[user2] and not((user2,item) in testing_set):            
                result.append((user2,sim))
      else:
          print(user,' not in neighbors ')

      

      result2=[]
      for (user2,sim) in result:
         n = NeighborInfo(user2,dict_uir[user2][item],sim) 
         result2.append(n)
    #   for i in result2:
    #     print(i.Neighbor_id,i.Rating,i.Similarity)
    #   input('press any key...')

      predicted = predict(ave_rating,user,result2[0:k])
      #print((user,item),'predicted:',predicted,'actual',dict_uir[user][item])
      
      prediction[(user,item)]=predicted


# %%
#14. 计算误差(N)
#-----------------------
mae=[]
for p in prediction:    
    predicted=prediction[p]
    user1,item1=p
    actual=dict_uir[user1][item1]    
    error = math.fabs(predicted-actual)
    #print(p,predicted,actual,error)
    mae.append(error)
#print(mae)
print('MAE:',sum(mae)/len(mae))


