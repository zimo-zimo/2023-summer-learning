# 实践III ：探究 LSTM 和 GRU 在 IMDB 数据集上性能

## 实践内容

- 查找资料学习 RNN，LSTM 和 GRU 模型
- 分别使用 pytorch 中提供的 LSTM 和 GRU 模块实现一个文本二分类模型，探究二者在 [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) 数据集上的性能表现
- 可直接使用 torchtext (https://pytorch.org/text/stable/datasets.html#imdb) 中提供的 IMDB dataset，词向量可采用 onehot 编码，也可以直接使用字符索引编码（可参考《动手深度学习》书第6章的相关内容）



## 实验结果

### LSTM

| Metric                | Accuracy |
| --------------------- | -------- |
| Accuracy for positive | 87.656%  |
| Accuracy for negative | 87.520%  |
| Total accuracy        | 88.020%  |


### GRU
| Metric                | Accuracy |
| --------------------- | -------- |
| Accuracy for positive | 85.600%  |
| Accuracy for negative | 88.624%  |
| Total accuracy        | 87.500%  |


**Parameters**
- embedding_dim = 100
- hidden_dim = 128
- num_layers = 2
- bidirectional = True
- dropout = 0.5
- batch_size = 64
- learning_rate = 0.001
- epochs = 30
- BCELoss()
- optim.Adam