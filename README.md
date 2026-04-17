# DWMF

Enviroment requirements are as follows:

| enviroment | version |
| ---------- | ------- |
| PyTorch    | 1.13.1  |
| CUDA       | 12.2    |
| Python     | 3.9.19  |

## About

The dataset includes three datasets: **weibo, twitter, gossipcop, and gossipcop-LLM**. According to the original author's intention, only the dataset link of **gossipcop-LLM** is given here:

- [https://github.com/junyachen/Data-examples](https://github.com/junyachen/Data-examples)

Other datasets can be obtained by contacting the original author.

The py file with the suffix of preprocess indicates the preprocessing of the dataset, and the py file with the suffix of dataset indicates the dataset class.

**train.py** is the training code. Running this code can get the reproducible results. Modifying the **forward** function in **class DWMF** can get different ablation experiment results.
