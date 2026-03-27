---
title: "FAISS (Facebook AI Similarity Search) 学习笔记"
description: "大规模向量相似度搜索、推荐系统、RAG (检索增强生成)、图像去重等。"
pubDate: "Mar 20 2026"
heroImage: "/blog-placeholder-4.jpg"
---


## 1. 什么是 FAISS？

**FAISS** (Facebook AI Similarity Search) 是由 Meta (原 Facebook) 开发的一个高效库，用于在海量向量数据中进行**相似性搜索**和**聚类**。

*   **核心优势**:
    *   **速度极快**: 比朴素线性扫描快几个数量级。
    *   **内存优化**: 支持向量压缩（量化），能在有限内存中存储数十亿向量。
    *   **GPU 加速**: 完美支持 NVIDIA GPU 加速索引构建和搜索。
    *   **算法丰富**: 提供从精确搜索到各种近似最近邻 (ANN) 算法的多种索引类型。

---

## 2. 核心概念

### 2.1 向量 (Vectors)
FAISS 处理的是固定维度的浮点数向量（通常是 `float32`）。这些向量通常由深度学习模型（如 BERT, ResNet, CLIP）提取的特征嵌入（Embeddings）。

### 2.2 距离度量 (Distance Metrics)
FAISS 主要支持两种距离计算：
*   **欧几里得距离 (L2)**: `METRIC_L2`，衡量空间直线距离。
*   **内积 (Inner Product)**: `METRIC_INNER_PRODUCT`，常用于余弦相似度（需先对向量归一化）。

### 2.3 索引 (Index)
索引是 FAISS 的核心对象，负责存储向量并执行搜索。不同的索引类型代表了不同的**精度 - 速度 - 内存**权衡策略。

---

## 3. 常用索引类型详解

### 3.1 暴力搜索 (Exact Search)
*   **名称**: `IndexFlatL2`, `IndexFlatIP`
*   **原理**: 线性扫描所有向量，计算距离并排序。
*   **特点**: 
    *   ✅ 100% 准确。
    *   ❌ 速度慢 ($O(N)$)，数据量大时不可用。
*   **适用**: 小数据集 (< 10万)，或作为其他索引的基准 (Ground Truth)。

### 3.2 倒排文件索引 (IVF - Inverted File System)
*   **名称**: `IndexIVFFlat`
*   **原理**: 
    1.  使用 K-Means 将向量聚类成 $nlist$ 个簇（Voronoi 区域）。
    2.  搜索时，只查询与目标向量最近的几个簇（`nprobe` 参数控制）。
*   **特点**: 
    *   ⚡ 速度显著提升。
    *   ⚖️ 牺牲少量精度换取速度。
*   **关键参数**: 
    *   `nlist`: 聚类中心数量（通常设为 $\sqrt{N}$ 的 4-16 倍）。
    *   `nprobe`: 搜索时探查的簇数量（越大越准，越慢）。

### 3.3 乘积量化 (PQ - Product Quantization)
*   **名称**: `IndexPQ`, `IndexIVFPQ`
*   **原理**: 
    *   将高维向量切分成 $M$ 个子向量。
    *   对每个子向量分别进行聚类量化，用较短的编码（如 8bit）表示。
    *   利用查表法快速估算距离。
*   **特点**: 
    *   💾 **极度节省内存**（可将 768 维 float32 压缩至 8-16 字节）。
    *   ⚡ 搜索极快。
    *   ⚠️ 有损压缩，精度下降较多。
*   **适用**: 十亿级向量库，内存受限场景。

### 3.4 标量量化 (SQ - Scalar Quantization)
*   **名称**: `IndexScalarQuantizer`
*   **原理**: 直接将浮点数转换为低位整数（如 8bit, 4bit）。
*   **特点**: 实现简单，压缩率不如 PQ，但在某些分布下表现稳定。

### 3.5 HNSW (Hierarchical Navigable Small World)
*   **名称**: `IndexHNSWFlat`, `IndexHNSWPQ`
*   **原理**: 基于图的导航算法，构建多层跳表结构。
*   **特点**: 
    *   🏆 **目前综合性能最好的索引之一**。
    *   无需训练（无需 K-Means），插入即可用。
    *   召回率高，速度快。
    *   内存占用相对较高（需存储图结构）。
*   **适用**: 对延迟敏感且追求高召回率的场景。

---

## 4. 快速上手代码示例 (Python)

### 环境安装
```bash
pip install faiss-cpu
# 如果有 GPU
pip install faiss-gpu
```

### 基础流程：训练 -> 添加 -> 搜索

```python
import faiss
import numpy as np

# 1. 准备数据
d = 128  # 向量维度
nb = 100000  # 数据库向量数量
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
# 模拟一些查询向量
nq = 10000
xq = np.random.random((nq, d)).astype('float32')

# 2. 创建索引 (以 IVFFlat 为例)
# METRIC_L2 表示欧几里得距离
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist=100, faiss.METRIC_L2)

# 3. 训练索引 (IVF 和 PQ 需要训练，Flat 和 HNSW 不需要)
print("Training index...")
index.train(xb)

# 4. 添加向量
print("Adding vectors...")
index.add(xb)

# 检查索引状态
print(f"Total vectors in index: {index.ntotal}")

# 5. 设置搜索参数
index.nprobe = 10  # 探查的簇数量，默认是 1，调大可提高准确率

# 6. 搜索
k = 5  # 返回最近的 5 个邻居
D, I = index.search(xq, k)

# 输出结果
print("Vector IDs of nearest neighbors:")
print(I[:5]) 
print("Distances:")
print(D[:5])
```

### HNSW 示例 (无需训练)

```python
# 创建 HNSW 索引
# M: 连接数 (越大越准，内存越大), efConstruction: 构建时的搜索深度
index_hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
index_hnsw.hnsw.efConstruction = 200 

# 直接添加
index_hnsw.add(xb)

# 搜索前设置 efSearch (运行时平衡速度与精度)
index_hnsw.hnsw.efSearch = 64

D, I = index_hnsw.search(xq, 5)
```

---

## 5. 索引选择指南 (决策树)

| 数据规模 | 内存限制 | 精度要求 | 推荐索引 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **< 100 万** | 无限制 | 100% | `IndexFlatL2` | 简单直接 |
| **< 100 万** | 无限制 | 高 | `IndexHNSWFlat` | 速度最快，无需训练 |
| **100 万 - 1 亿** | 中等 | 高 | `IndexIVFFlat` | 需训练，调节 `nprobe` |
| **100 万 - 1 亿** | 中等 | 中高 | `IndexHNSWPQ` | 结合图与量化 |
| **> 1 亿** | 严格 | 中 | `IndexIVFPQ` | 经典的大规模方案 |
| **> 1 亿** | 严格 | 中 | `IndexPQ` | 极致压缩 |

---

## 6. 高级技巧与最佳实践

1.  **归一化 (Normalization)**:
    *   如果使用余弦相似度，务必在存入 FAISS 前对向量进行 L2 归一化 (`faiss.normalize_L2`)，并使用 `METRIC_INNER_PRODUCT`。这通常比直接用 `COSINE` 度量更快。
    
2.  **复合索引 (Index Shards / Replicas)**:
    *   **Shards**: 将数据分片存储在不同索引中，并行搜索后合并结果（适合超大数据集）。
    *   **Replicas**: 复制多份索引到不同 GPU，负载均衡。

3.  **持久化 (Serialization)**:
    *   使用 `faiss.write_index(index, "my_index.faiss")` 保存。
    *   使用 `faiss.read_index("my_index.faiss")` 加载。
    *   *注意*: 对于包含训练数据的索引（如 IVF, PQ），必须先训练再保存。

4.  **参数调优**:
    *   **IVF**: `nlist` 通常设为 $4 \times \sqrt{N}$ 到 $16 \times \sqrt{N}$。`nprobe` 越大越准，建议从 $\sqrt{nlist}$ 开始尝试。
    *   **HNSW**: `efConstruction` 影响建库时间，`efSearch` 影响查询时间。

5.  **GPU 加速**:
    *   可以将索引移动到 GPU: `res = faiss.StandardGpuResources(); gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)`。
    *   适合批量搜索或构建超大索引。

---

## 7. 常见问题 (FAQ)

*   **Q: FAISS 支持动态增删吗？**
    *   A: 支持 `add`，但不支持高效的单条 `remove`。如果需要频繁删除，通常采用“标记删除”或定期重建索引的策略。HNSW 支持部分删除操作但效率一般。
*   **Q: 为什么搜索结果为空或距离异常大？**
    *   A: 检查向量维度是否匹配，数据类型是否为 `float32`，以及是否忘记训练（针对 IVF/PQ）。
*   **Q: 如何在分布式环境下使用？**
    *   A: FAISS 本身是单机库。分布式通常需要结合 Spark (Spark-Faiss) 或在应用层做分片路由（Sharding）。

---
