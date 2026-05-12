# SimpleIMDBClassifier
这是一个简易评论分类器，使用[IMDB数据集](https://ai.stanford.edu/~amaas/data/sentiment/)训练。  
这并不是一个实际项目，更多基于学习用途。  

# 运行环境
程序基于以下环境运行：
```bash
datasets                         4.8.5
numpy                            1.26.2
python                           3.11.5
torch                            2.4.1+cu124
```
除了为pytorch提供GPU加速的cuda，其他库都可以使用pip安装。cuda并不是必须的。  

### 潜在问题
脚本在命令行下运行时可能存在问题。运行环境实际上包括IPython Console。  
```bash
ipykernel                        6.29.5
ipython                          8.27.0
```

# 使用
```torch_train.py```基于pytorch训练分类器模型。一旦训练完成，```torch_run.py```可以使用最新的检查点恢复模型并提供命令行IO和模型交互。  
```numpy_train.py```尝试自行模拟自注意力机制，不涉及pytorch，也没有任何模型保存。使用的数据量比torch版本更少。不过值得注意的是，模型存在问题，并不能很好学习数据。  

# 杂项
### ISSUE & PR
这个项目不会得到维护。不接受任何ISSUE/PR。

### LICENSE
[MIT LICENSE](LICENSE)  
