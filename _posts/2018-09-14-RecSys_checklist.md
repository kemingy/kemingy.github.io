---
title: Recommendation System Checklist
categories: [Note, Deep Learning, NLP]
---

Notes of ["36 strokes of recommended system"](https://time.geekbang.org/column/74)

<!-- more -->

## Basic

### When to use?

$\frac{N_{connection}}{N_{user} \times N_{item}}$

### Stage

1. mining
2. recall
3. ranking

### Forecast

* score
  * collect data
  * quality of score
  * unstable of score
* action
  * Click Through Rate(CTR)
  * dense of data
  * implicit feedback

### Problem

* cold start
* exploit and explore
* secure

### User profile

* Vectorization
* based on records, statistics, balck box models
* Feature
  * TF-IDF, TextRank
  * Named Entity Recognition (based on dictionary and conditional random field)
  * Text classification
  * Clustering
  * Topic model
  * Embedding
* Select labels
  * Chi-square test(CHI)
  * Information gain(GI)

## Models

### Content-based

* friendly to new items and new users(cold start)
* easy to get data
  * grab (use crawler)
  * clean data
  * data mining
  * compute
* Algorithm
  * Unsupervised
    * BM25
    * cosine similarity
  * Supervised
    * GBDT
    * Logistic Regression

### Collaborative filtering

* Memory-based
  * User-based
    * sample to reduce vector dimension(DIMSUM)
    * Map-Reduce to get user2user similarity, or use (KGraph, GraphCHI)
    * punish hot items
    * attenuate with time
    * cons
      * too many users to calculate
      * too sparse to get real similar users(hot items)
  * Item-based
    * normalization for users and items
    * Slope one(consider confidence)
  * cons: sparse => vibrate to small correlations
* Model-based
  * Singular Vector Decomposition(SVD): $$min_{q^*,p^*}\Sigma_{(u,i)\in \kappa}(r_{ui}-p_u q_i^T)^2+\lambda(\lVert q_i\rVert ^2+\lVert p_u\rVert ^2)$$
  * add bias: $$\hat{r}_{ui}=\mu + b_i + b_u + p_u q_i^T$ , $min_{q^*,p^*}\Sigma_{(u,i)\in \kappa}(r_{ui}-\mu -b_i - b_u -p_u q_i^T)^2+\lambda(\lVert q_i\rVert ^2+\lVert p_u\rVert ^2+b_i^2+b_u^2)$$ 
  * SVD++(add implicit feedback & user attribute): $$\hat{r}_{ui}=\mu + b_i + b_u + (p_u+\lvert N(u) \rvert ^{-0.5}\Sigma_{i\in N(u)}x_i+\Sigma_{a\in Au}y_a)q_i^T$$
  * consider about time
    * weight
    * time scope
    * special date
  * cons:
    * ranking is what we want, not vectors
    * negative sampling is still a problem

### Similarity

* Euclidean distance: $d(p, q)=\sqrt{\Sigma_{i=1}^n(q_i-p_i)^2}$

* Cosine similarity: $cos(\theta)=\frac{A\cdot B}{\lVert A\rVert \lVert B\rVert}$
  * adjust: $sim(i,j)=\frac{\Sigma_{u\in U}(R_{u,i}-\bar{R_u})(R_{u,j}-\bar{R_u})}{\sqrt{\Sigma_{u \in U}(R_{u,i}-\bar{R_u})^2}\sqrt{\Sigma_{u \in U}(R_{u,j}-\bar{R_u})^2}}$
* Pearson correlation: $\rho_{X,Y}=\frac{\Sigma^n_{i=1}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\Sigma^n_{i=1}(x_i-\bar{x})^2}\sqrt{\Sigma^n_{i=1}(y_i-\bar{y})^2}}$
* Jaccard index: $J(A, B)=\frac{A\bigcap B}{A\bigcup B}$

### Optimization

* Stochastic Gradient Descent(SGD)
* Alternative Least Square(ALS): $R_{m\times n}=P_{m\times k}\times Q^T_{n\times k}$
  * easy to parallelize
  * fast than SGD in not too sparse data
  * Weighted-ALS: one-class    $$min_{q^*,p^*}\Sigma_{(u,i)\in \kappa}c_{ui}(r_{ui}-p_uq_i^T)^2+\lambda(\lVert q_i\rVert ^2+\lVert p_u\rVert ^2)$$ where $c_{ui}=1+\alpha n$ stands for confidence and  $\alpha=40$ , $n$ is frequence.
    * sample in hot items (negative sampling)
  * Tools to search similar items: Faiss(ball tree), Annoy, NMSlib, KGraph

### Ranking

* methods: point-wise, pair-wise, list-wise
* Bayes Personalized Recommendation(BPR)
  * sample: (user, item1, item2, True/False)
  * $\Pi_{u,i,j}p(i>_uj\mid \theta)p(\theta)$
  * Mini-batch Stochastic Gradient Descent(MSGD)
* Area Under Curve(AUC): $AUC=\frac{\Sigma_{i\in samples}r_i-\frac{1}{2}M\times (M-1)}{M\times N}$ 

### Ensemble

* Logistic Regression
* Follow-the-Regularized-Leader(FTRL)
* Gradient Boost Decision Tree(GBDT)
* Factorization Machine(FM)
  * $\hat{y}=w_0+\Sigma^n_{i=1}w_ix_i+\Sigma^n_{i=1}\Sigma^n_{j=i+1}<V_i,V_j>x_ix_j$ 
  * $\sigma(\hat{y})=\frac{1}{1+e^{-\hat{y}}}$ , $loss(\theta)=-\frac{1}{m}\Sigma^m_{i=1}[y^{(i)}log(\sigma(\hat{y}))+(1-y^{(i)})log(1-\sigma(\hat{y}))]$ 
  * $\Sigma_{i=1}^n\Sigma_{j=i+1}^n<v_i,v_j>x_ix_j$ = $\frac{1}{2}\Sigma_{f=1}^k((\Sigma_{i=1}^nv_{i,f}x_i)^2-\Sigma_{i=1}^nv_{i,f}^2x_i^2)$ 
* Field-aware Factorization Machine(FFM)
  * $\hat{y}=w_0+\Sigma^n_{i=1}w_ix_i+\Sigma^n_{i=1}\Sigma^n_{j=i+1}<V_{i,fj}, V_{j,fi}>x_ix_j$ 
* Deep&Wide
  * $P(Y=1\mid X)=\sigma(W^T_{wide}[X, \Phi(X)]+W^T_{deep}a^{(l_f)}+b)$ 	
  * scale: $\frac{i-1}{n_q-1}$
  * model
    * deep: full-connected layer and ReLU
    * wide: cross combination
  * deploy

### Bandit

* Multi-armed bandit problem(MBP): cold start and EE problem
* $R_T=\Sigma_{i=1}^T(w_{opt}-w_{B(i)})=Tw^*-\Sigma^T_{i=1}w_{B(i)}$ 
* Thompson sampling
  * beta distribution
    * $\alpha + \beta$ ↗⇒ center
    * $\frac{\alpha}{\alpha+\beta}$ ↗⇒ center → 1
  * `choice = numpy.argmax(pymc.rbeta(1 + self.wins, 1 + self.trials - self.wins))`
* Upper Confidence Bound(UCB)
  * $\bar{x}_j(t)+\sqrt\frac{2\ln t}{T\_{j, t}}$
  * $t$ is total times, $\bar{x}$ is average gain
* Epsilon greedy
  * select $\epsilon\in(0, 1)$, select best chioce with $p_{best}=1-\epsilon$ 
* LinUCB
  * add features, support delete items
  * contextually related
  * expected revenue: $\hat{r}=x^T_{d\times 1}\hat{\theta}_{d\times 1}$
  * upper bound: $\hat{b}=\alpha \sqrt x^T_{d\times 1}(D^T_{m\times d}D_{m\times d}+I_{d\times d})^{-1}x_{d\times 1}$ 
* COFIBA
  * use CF to reduce user group
  * update item group and user group with LinUCB

### Deep learning

* Product2vec
  * users' browser history == docs
  * try to learn vector of items
* Youtube
  * $P(w_t=i\mid U,C)=\frac{e_{v_iu}}{\Sigma_{j\in V}e_{v_ju}}$ 
  * embedded watches, search tokens, geographic, age, gender......
* Spotify
  * RNN: $h_t=F(Wx_t+Uh_{t-1})$

### Leaderboard

* Hacker News: $\frac{P-1}{(T+2)^G}$
  * $P$: vote, $T$: time(hour), G: gravity
* Newton's law of cooling: $T(t)=H+Ce^{-\alpha t}$
  * $H$: environment tempurature,  $C$: vote, $\alpha$: Chill coefficient
* StackOverflow: $\frac{(log_{10}Q_{views}\times 4 + \frac{Q_{answers}\times Q_{score}}{5}+\Sigma_iA_{score_i})}{(\frac{Q_{age}}{2}+\frac{Q_{update}}2+1)^{1.5}}$ 
  * $Q_{age}$: time, $Q_{update}$: last update time
* Wilson section: $\frac{\hat{p}+\frac{1}{2n}z^2\_{1-\frac{\alpha}{2}}\pm z\_{1-\frac{\alpha}2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2\_{1-\frac{\alpha}2}}{4n^2}}}{1+\frac{1}nz^2\_{1-\frac{\alpha}2}}$ 
  * $\hat{p}$: praise rate, $z_{1-\frac{\alpha}2}$: Z statistic with confidence $\alpha$ 
* Bayes average: $\frac{v}{v+m}R+\frac{m}{v+m}C$ 
  * $R$: average score, $v$: vote, $m$: average vote, $C$: average score

### Weighted sampling

* limit data
  * $S_i=R^{\frac{1}{w_i}}$
  * $f(x,\lambda)=\begin{cases} \lambda e^{-\lambda x} & \text{if }x>0 \\ 0 &\text{if }x\leqslant 0 \end{cases}$
* unlimit data
  * keep $k$ items, replace item with $p=\frac{k}n$ 

### Deduplicated

* SimHash
* Bloom filter

## Data

### Collect data

* data model
  * user profile
  * item profile
  * relation
  * event
* store
  * buried point: SDK, Google Analytics, Mixpanel
  * end: front-end, back-end
* architecture
  * Nginx/server → logstash/flume → kafka → storm → HDFS

## Defence

### Attack methods

* item: target, mask, fake
* score: random, average

### Defence

* platform level: sign up cost, reCAPTCHA
* data level: antispam(PCA, classify, cluster)
* algorithm level: add user quality, restrict user weight

## Deploy

### Real Time

* response → update feature → update model

* Storm
  * Spout, Bolt, Tuple, Topology
* Kafka
* Hoeffding
  * the real $E(x)$ is smaller than $\hat{x}+\epsilon$ with probability $p=1-\delta$ : $\epsilon=\sqrt{\frac{ln(1/\delta)}{2n}}$ 
* Sliding window
* Sampling

### Test Platform

* scale: $N>=10.5(\frac{s}{\theta})^2$ , $s$ is standard deviation, $\theta$ is sensitivity. (90% confidence)
* Google platform:
  * A domain is a segmentation of traffic
  * A layer corresponds to a subset of the system parameters
  * An experiment is a segmentation of traffic where zero or more system parameters can be given alternate values that change how the incoming request if processed.

### Database

* Offline and Online
* Choice
  * feature, model, result
  * Inversed index: Redis or Memcached
  * Column-oriented DBMS: HBase, Cassandra
  * Store and search: ElasticSearch, Lucene, Solr
  * Parameters: PMML

### API

* recommend_ID: unique ID, trace the performance

## Tools

### CF and MF

* KNN
  * kgraph, annoy, faiss, nmslib
* SVD, SVD++, BPR
  * lightfm
* ALS
  * implicit, QMF

### Ensemble

* GBDT:
  * LightGBM, XGBoost
* FM, FFM:
  * libffm
* Linear: vowpal_wabbit
* Wide&Deep Model

### All in one

* PredictionIO
* recommendationRaccoon
* easyrec
* hapiger
