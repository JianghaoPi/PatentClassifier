# 专利顶层分类器--工科创Ⅳ-J大作业

+ 该项目要求使用如下三种方式对专利进行分类：
>1. 常规单线程模式，用liblinear对其进行分类
>2. 最小最大模块化网络多线程模式，先将原分类问题随机分解，用liblinear对子问题分类后再用最小最大模块化网络对将子问题的解组合起来
>3. 最小最大模块化网络多线程模式，先基于先验知识（层次化标号结构信息）将原分类问题随机分解，用liblinear对子问题分类后再用最小最大模块化网络对将子问题的解组合起来
+ 考虑到该项目需要并行计算，而Java内置多线程很强大，所以使用了liblinear库的Java版本，删去了训练和预测的Train.java和Predict.java，并参考其实现写出了自己的训练和预测java类，封装了所有的需要训练和预测的函数，同时也针对本项目的数据特点对liblinear原有的一些类也进行了适当修改。

## 工程结构介绍

### PatentClassifier.java
这是整个工程的入口类，包含了Basic(),RandomMinMaxModule(),LabeledMinMaxModule()以及唯一主函数main()。
+ Basic()实现了1中的要求，即常规单线程学习该两类问题
+ RandomMinMaxModule()实现了2中的要求，即随机方式分解原问题，多线程用libliear学习每个子问题，并使用最小最大模块化网络将子问题的解合起来
+ LabeledMinMaxModule()实现了3中的要求，即基于先验知识分解原问题，多线程用libliear学习每个子问题，并使用最小最大模块化网络将子问题的解合起来

### Linear.java
这是liblinear的核心代码，参考其中的训练和预测函数，添加了patentPredictValue()函数，用起来更方便

### MyTrain.java
该类包含三种训练方式的所有训练函数

### MyPredict.java
该类包含两种预测方式，因随机分解和基于先验知识分解本质上都是先分解进行训练再将解组合起来，所以预测函数完全一样

### MyPerformanceEvaluation.java
该类是用于评估结果，完成4中的要求，用于计算ROC曲线点的坐标以及F1的值，两种最大最小模块化网络实现的评估方式一样

### MyUtil.java
该类是一个工具类，包含常用的类和函数