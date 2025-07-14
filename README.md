# ViT简明解析
## 目录
1. ViT核心原理概述
---
## ViT核心原理概述
ViT通过将图像分成一系列的**图块（patches）**，并将每个图块转换为**向量**表示作为输入序列。然后，这些向量将通过**多层的Transformer编码器**进行处理，其中包含了**自注意力机制**和**前馈神经网络层**。这样可以捕捉到图像中不同位置的上下文依赖关系。最后，通过对Transformer编码器输出进行分类或回归，可以完成特定的视觉任务。  
好的，我们来详细拆解一下 Vision Transformer (ViT) 的具体结构。ViT 的核心思想是 **将标准的 Transformer 架构（原本为 NLP 设计）直接应用于图像分类任务**，摒弃了传统卷积神经网络 (CNN) 的卷积和池化操作。

## ViT具体结构
![ViT 结构示意图](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

**1. 图像分块嵌入 (Patch Embedding)**

*   **目的：** 将二维图像数据转换为 Transformer 能够处理的一维序列数据。
*   **过程：**
    *   **输入图像：** `H x W x C` (高度 x 宽度 x 通道数，例如 224x224x3)。
    *   **分块：** 将图像分割成固定大小 `P x P` 的小块（称为 Patches）。通常 `P = 16`，那么一个 224x224 的图像会被分割成 `(224/16) x (224/16) = 14 x 14 = 196` 个图像块。
    *   **展平：** 将每个 `P x P x C` 的图像块展平成一个长度为 `P² * C` 的向量。对于 `P=16, C=3`，每个向量长度就是 `16*16*3 = 768`。
    *   **线性投射 (Linear Projection)：** 通过一个可学习的线性层（全连接层）将这些展平后的向量映射到一个更高维的嵌入空间 `D`（称为 Embedding Dimension 或 Hidden Size，例如 `D=768`）。这个线性层通常被称为 `Patch Embedding Projection`。
    *   **输出：** 得到一个形状为 `N x D` 的张量。其中 `N = (H*W) / P²` 是图像块的数量（序列长度），`D` 是嵌入维度（每个块的向量表示）。上例中就是 `196 x 768`。

**2. 添加 [class] Token (Prepend Class Token)**

*   **目的：** 为整个图像提供一个全局的、可学习的表示，用于最终的分类任务。
*   **过程：**
    *   创建一个可学习的嵌入向量（`1 x D`），称为 `[class] token` 或 `x_class`。
    *   将这个 `[class] token` **前置** 到第 1 步得到的 Patch Embedding 序列的开头。
    *   **输出：** 序列长度变为 `N+1`。上例中就是 `(196 + 1) x 768 = 197 x 768`。这个额外的 token 在 Transformer 处理过程中会与其他所有图像块 token 进行交互，最终它的状态将作为整个图像的表示被送到分类头。

**3. 添加位置编码 (Add Positional Embedding)**

*   **目的：** 为序列中的每个 token（包括图像块 token 和 [class] token）注入空间位置信息。因为 Transformer 本身是**置换等变 (Permutation Equivariant)** 的，它对输入序列的顺序不敏感，但图像的空间位置信息对于理解图像内容至关重要。
*   **过程：**
    *   创建一个可学习的矩阵（或使用固定的如正弦编码）`E_pos`，其形状为 `(N+1) x D`。每一行对应序列中的一个位置（包括 [class] token 的位置 0 和 N 个图像块的位置 1 到 N）。
    *   将这个位置编码 `E_pos` **按元素相加** 到第 2 步得到的 `(N+1) x D` 的 token 序列上：`z_0 = [x_class; x_p1; x_p2; ...; x_pN] + E_pos`。
    *   **输出：** `z_0` 的形状仍然是 `(N+1) x D`。现在每个 token 向量都包含了其原始图像内容信息（通过 Patch Embedding）和其在图像中的位置信息（通过 Positional Embedding）。`z_0` 就是 Transformer Encoder 的初始输入。

**4. Transformer Encoder (Stacked Encoder Blocks)**

*   **目的：** 对包含空间信息的 token 序列进行深层次的表示学习和特征提取。序列中的每个 token 都通过自注意力机制与序列中所有其他 token 进行交互，捕获全局的上下文依赖关系。
*   **结构：** ViT 使用标准的 **Transformer Encoder** 结构，由 `L` 个相同的 **Encoder Block** 堆叠而成（例如 `L=12`）。每个 Encoder Block 包含两个核心子层：
    *   **a. 多头自注意力层 (Multi-Head Self-Attention - MSA):**
        *   输入：来自上一层的序列 `z_l` (`(N+1) x D`)。
        *   核心操作：每个 token 生成 Query (Q), Key (K), Value (V) 向量（通过线性变换）。
        *   计算注意力分数：`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`，其中 `d_k` 是 Key 的维度。
        *   **“多头” (Multi-Head)：** 将 `D` 维的 Q, K, V 投影到 `h` 个（例如 `h=12`）不同的、维度为 `d_k = d_v = D/h` 的子空间（称为 Head）。在每个 Head 上独立计算注意力，然后将 `h` 个 Head 的输出拼接起来，再通过一个线性层投影回 `D` 维。
        *   **自注意力 (Self-Attention)：** Q, K, V 都来自于同一个输入序列 `z_l`。这使得每个 token 都能关注序列中所有其他 token（包括它自己），从而捕获全局的上下文信息。
        *   输出：`MSA(z_l)` (`(N+1) x D`)。
    *   **b. 多层感知机层 (Multi-Layer Perceptron - MLP):**
        *   输入：MSA 层的输出经过 LayerNorm 和残差连接后的结果。
        *   结构：通常是两个全连接层，中间夹着一个非线性激活函数（如 GELU）。第一个全连接层将维度扩展到 `4*D`（或其它比例），第二个全连接层再压缩回 `D`。即 `MLP(x) = FC2(GELU(FC1(x)))`。
        *   目的：对每个 token 的特征进行非线性变换和增强。
        *   输出：`MLP(x)` (`(N+1) x D`)。
*   **每个 Encoder Block 内的完整流程：**
    1.  **LayerNorm 1:** `z_l' = LayerNorm(z_l)`
    2.  **MSA:** `msa_out = MSA(z_l')`
    3.  **残差连接 1:** `z_msa = z_l + msa_out` (保持信息流，缓解梯度消失)
    4.  **LayerNorm 2:** `z_msa' = LayerNorm(z_msa)`
    5.  **MLP:** `mlp_out = MLP(z_msa')`
    6.  **残差连接 2:** `z_{l+1} = z_msa + mlp_out`
*   **堆叠：** 将上述 Block 重复 `L` 次：`z_l = EncoderBlock(z_{l-1})` for `l = 1 ... L`。
*   **输出：** `z_L` (`(N+1) x D`)，这是经过 `L` 层 Transformer Encoder 深度处理后的 token 序列表示。

**5. MLP 分类头 (MLP Head / Classifier)**

*   **目的：** 利用序列中第一个 token（即 `[class] token`）的最终状态表示整个图像，并预测其所属的类别。
*   **过程：**
    *   **提取 [class] token：** 从 Transformer Encoder 的输出 `z_L` (`(N+1) x D`) 中取出第一个位置（索引为 0）的向量 `z_L^0` (`1 x D`)。这个向量已经融合了整个图像所有块的信息。
    *   **层归一化：** 通常会对 `z_L^0` 应用一个额外的 LayerNorm：`y = LayerNorm(z_L^0)` (`1 x D`)。
    *   **MLP 分类器：** 将 `y` 输入到一个小的 MLP（通常是一个或两个隐藏层的全连接网络）。
        *   最简单的形式：一个线性层（无隐藏层）：`logits = Linear(y)` (`1 x num_classes`)。
        *   更常见的形式：`Linear(GELU(Linear(y)))`。第一个线性层通常将维度映射到与 Embedding Dimension `D` 相同或更小，第二个线性层映射到类别数 `num_classes`。
    *   **输出：** `logits` (`1 x num_classes`)，表示图像属于每个类别的未归一化分数。
    *   **最终预测：** 对 `logits` 应用 Softmax 函数得到概率分布，取概率最大的类别作为预测结果。

**关键设计要点总结：**

1.  **序列化：** 通过分块嵌入将图像转换为序列。
2.  **[class] Token：** 提供一个用于分类的全局图像表示锚点。
3.  **位置编码：** 显式注入至关重要的空间位置信息。
4.  **标准 Transformer Encoder：** 核心计算单元，利用自注意力捕获全局依赖关系。MSA 和 MLP 子层 + LayerNorm + 残差连接是其标准构成。
5.  **分类头：** 仅使用 `[class] token` 的最终状态进行分类预测。

**ViT 的优势与挑战：**

*   **优势：** 强大的全局建模能力、避免局部归纳偏置（理论上能更好地学习长距离依赖）、结构相对统一、在**大规模数据集**上训练时性能超越顶尖 CNN。
*   **挑战：** 需要大量数据预训练（在小数据集上容易过拟合）、计算复杂度随序列长度平方增长（处理高分辨率图像开销大）、缺乏 CNN 固有的局部性和平移等变性偏置（需要更多数据和位置编码来学习）。

## 参考资料
[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929)  
[ViT（Visual Transformer）最通俗易懂的讲解（有代码）](https://blog.csdn.net/2301_77653781/article/details/142360725)  
[VIT （Vision Transformer）深度讲解](https://www.bilibili.com/video/BV15RDtYqE4r?vd_source=a148a1dc47c921e008b6a45f395d3cf0)  
[【Transformer系列】深入浅出理解ViT(Vision Transformer)模型](https://blog.csdn.net/m0_37605642/article/details/133821025)  
