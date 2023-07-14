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


Parameters
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