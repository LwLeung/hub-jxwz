import jieba # 与字典做最长前缀匹配划分单词
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model # 线性模型模块
from sklearn import tree # 决策树模块


dataset = pd.read_csv("dataset.csv",sep="\t",header=None)


# 第一列做分词
input_sentence = dataset[0].apply(lambda x:" ".join(jieba.lcut(x))) #sklearn本身不支持中文，这一步是为了让sklearn能对中文做处理
# print(input_sentence)

vector = CountVectorizer() # 实例化
vector.fit(input_sentence.values) # 提取文本特征
input_feature = vector.transform(input_sentence.values) # 转换成特征向量

test_queries = ["今天是什么天气","帮我放一首轻松的音乐","导航到中央公园","去我闺蜜家怎么走？","打开所有灯"]
for i in range(len(test_queries)):
    test_query = test_queries[i]
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print("待预测的文本", test_query)

    # 使用knn模型，进行文本分类操作
    for k in [1,3,5,7,9]:
        model = KNeighborsClassifier(n_neighbors = k) 
        model.fit(input_feature, dataset[1].values) # 基于特征向量和标签训练模型
        print(str(k)+"-KNN模型预测结果: ", model.predict(test_feature))

    # 使用线性模型，进行文本分类操作
    model = linear_model.LogisticRegression(max_iter=1000) # 模型初始化
    model.fit(input_feature, dataset[1].values)
    print("线性模型预测结果: ", model.predict(test_feature))

    # 使用决策树模型，进行文本分类操作
    model = tree.DecisionTreeClassifier() # 模型初始化
    model.fit(input_feature, dataset[1].values)
    print("决策树模型预测结果: ", model.predict(test_feature))

    print("\n")
