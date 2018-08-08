---
title: 'Relevance: from TFIDF to BM25'
date: 2018-03-12 23:40:50
tags: Elasticsearch
categories: [技术, 笔记]
---

在信息检索系统中，TFIDF 和 BM25 函数都可以用来计算文档的相关度并进行排序。其中，TFIDF 也经常被用在自然语言处理中，BM25 则可以看做是 TFIDF 的进一步演化。在处理搜索字符串的时候，它们采用的都是 Bag-of-Word 方式，也就是说会忽略掉词的出现顺序，只考虑词出现与否和词频。

<!-- more -->

## TFIDF

TFIDF 分为 TF (Term Frequency) 词频和 IDF (Inverse Document Frequency) 逆文档频率。

试想一下，我们想计算一个词的重要性，最自然的想法就是看看这个词在文章中出现了多少次，频繁出现的词很可能跟文章讨论的话题相关，这也就是计算 TF 的理由。另一方面，stop words 也具有出现频率非常高的特性，为了避免这些词影响我们对文章关键词的分析，就需要用到逆文档频率，也就是查看这些词在其他文档中出现的频率，频率越高说明该词越普通，很可能属于 stop words 而不是我们想找的文章关键词。

TF 和 IDF 公式都有很多变种，这里介绍最常见的。

$$TF(t, d) = f_{t, d}/\sum_{t' \in d}f_{t', d}$$

$$IDF(t, D)=log(\frac{N}{\{d \in D : t \in d\}})$$

$$TFIDF(t, d, D) = TF(t, d) \cdot IDF(t, D)$$

其中，$t$ 表示一个词，$d$ 表示一篇文档，$D$ 表示文档的集合，$N$ 表示文档的总数。

在 ES (Elasticsearch) 中，通常是取 $sqrt(TF)$ ，并对 $IDF$ 进行平滑处理并加 1，变成 $log(\frac{N}{(d \in D : t \in d) + 1}) + 1$ 。此外，ES 中还会考虑匹配到的文本长度，短文本中出现的关键词明显要比长文本中的关键词更有说服力，引入 $Norm= 1 / sqrt(length)$ ，最终公式为 $TFIDF_{ES} = TF_{ES} \cdot IDF_{ES} \cdot Norm$ 。值得注意的是，ES 中计算文档总数的时候，会把刚刚删除的文档也算进去，进行合并之后才会正常，而且 $Norm$ 的值是以 8-bit float 型保存的，因此经常会出现很多奇怪的问题。

## BM25

BM 表示 best match 。BM25 经常被用作搜索引擎的匹配度排序函数，目前是 Lucene, Solr, Elasticsearch 的默认函数。

BM25 的公式如下：

$$IDF(q_i) = log \frac{N-n(q_i)+0.5}{n(q_i)+0.5}$$

$$BM25(D, Q) = \sum_{i=1}^{n}IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{\|D\|}{avgdl})}$$

其中，$f(q_i, D)$ 表示 $q_i$ 在 $D$ 中的频率，$\|D\|$ 表示 $D$ 中词的个数，$avgdl$ 表示文档的平均长度，$k_1$ 和 $b$ 是两个参数，$k_1 \in [1.2, 2.0]$ 调节词频结果在词频饱和度中的增长速度，$b \in [0, 1.0]$ 调节文本长度对词的归一化程度。

和 TFIDF 相比，BM25 最大的特点就是增加了文本长度对词的归一化程度，并且这个归一化比上面提到的 ES 中 TFIDF 的归一化要更加合理，考虑到了所有文档的平均长度。

在 ES 的实现中，$IDF$ 部分跟 TFIDF 一样增加了平滑处理并加 1，因为原来的式子有可能出现负值。

## 数值分析

### IDF

[![tf]({{ "assets/img/tf.png" | relative_url }})](https://www.desmos.com/calculator/hg5tuporhs)

红色为经典的 BM25 IDF 函数曲线，蓝色为 ES 中使用的 IDF 曲线。（N=200）

从图中可知，函数的极值跟 N 有关，N 越大，极值越大。随着 $x$ 的增大，IDF 值减小的速度逐渐放缓。

### TF

首先，令 $b = 0$ ，观察函数的走势。

[![tf]({{ "assets/img/tf.png" | relative_url }})](https://www.desmos.com/calculator/i700nwzj68)

红色为 ES 中 TFIDF 的 TF 曲线，蓝色为 ES 中 BM25 的 TF 曲线。TFIDF 的 TF 是没有上界的，而 BM25 的 TF 上界为 $k_1$ ，通常取 $k_1 = 1.2$ 。也就是说，到达一定频率后，再增加也不会对结果产生太大影响。

之后考虑不同的文本长度对函数的影响。

[![tfd]({{ "assets/img/tfd.png" | relative_url }})](https://www.desmos.com/calculator/p9babkz6p8)

三条线从上到下依次是平均文档长度的 0.1，1，10 倍，取 $k_1 = 1.2, b = 0.75$ 。三条线趋于饱和的速度跟平均文档的长度相关，平均文档长度越长，趋于饱和的速度越慢。也就是说，如果是长文本的搜索，需要较高的词频才能使 TF 值接近饱和，这一点比较符合人们的认知。

综上所述，BM25 的极值取决于文档总数 $N$ 和参数 $k_1$，即 $max(BM25) = k_1 \cdot log(N)$ 。

## 局限性

BM25 相对于 TFIDF 有了很大的改进，有的人认为 BM25 是 TFIDF 类函数的 state-of-the-art 。客观来说，在文档搜索方面，BM25 取得了很好的效果，但也仅限于对文档进行搜索匹配。搜索是一项复杂的任务，在某些场景下，可能更需要选取合适的函数，或者采取其他策略。