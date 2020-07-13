# 版本 0407

## 模型

- 输入：一维原始信号【3072，2】

- 输出：一维信号【3072，2】

- 网络结构：Auto Encoder ， 【CONV 1D, BatchNorm, ReLU】+【DeCONV 1D】 ，输出层 tanh

- 网络参数：

  ```python
  N_LAYERS = 17
  FILTER_SIZE = 7
  STRIDE = 4
  DEPTHS = [32, 64, 128, 256, 256]
  ```

- 训练参数：

  ```json
  {
    "CUDA_VISIBLE_DEVICES": "4,5"
  }
  ```

  