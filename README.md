# ViT简明解析
## 目录
1. ViT核心原理概述
---
## ViT核心原理概述
ViT通过将图像分成一系列的**图块（patches）**，并将每个图块转换为**向量**表示作为输入序列。然后，这些向量将通过**多层的Transformer编码器**进行处理，其中包含了**自注意力机制**和**前馈神经网络层**。这样可以捕捉到图像中不同位置的上下文依赖关系。最后，通过对Transformer编码器输出进行分类或回归，可以完成特定的视觉任务。
## 参考资料
[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929)
[ViT（Visual Transformer）最通俗易懂的讲解（有代码）](https://blog.csdn.net/2301_77653781/article/details/142360725)
[VIT （Vision Transformer）深度讲解](https://www.bilibili.com/video/BV15RDtYqE4r?vd_source=a148a1dc47c921e008b6a45f395d3cf0)
[【Transformer系列】深入浅出理解ViT(Vision Transformer)模型](https://blog.csdn.net/m0_37605642/article/details/133821025)
