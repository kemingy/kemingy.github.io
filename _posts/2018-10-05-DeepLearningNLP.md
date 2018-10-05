---
title: NLP with Deep Learning
categories: [笔记, Deep Learning, NLP]
---

Notes of [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

<!-- more -->

## Word Embedding

$J(\theta)=-\frac{1}{T}\Sigma^T_{t=1}\Sigma_{-m\le j \le m}\log P(w_{t+j}\vert w_t;\theta)$

We will use 2 vectors per word $w$ to calculate $P(w_{t+j}\vert w_t;\theta)$

1. $v_w$ when $w$ is a center word
2. $u_w$ when $w​$ is a context word

$P(o\vert c)=\frac{\exp(u_o^{\top}v_c)}{\Sigma_{w\in V}\exp(u^{\top}_wv_c)}$

> Why two vectors?
>
> Easier optimization. Average both at the end.

Two model variants:

1. Skip-grams(SG)

   Predict context words (position independent) given center word.

2. Continuous Bag of Words(CBOW)

   Predict center word from (bag of) context words.

Additional efficiency in training: negative sampling.

$arg\max J(\theta)=\frac 1 T\Sigma^T_{t=1}J_t(\theta)$

$J_t(\theta)=\log \sigma(u_o^{\top}v_c)+\Sigma_{i=1}^k \mathbb{E}_{j\sim P(w)}[\log \sigma(-u^{\top}_jv_c)]$

$\sigma(x) = \frac 1 {1+e^{-x}}$

$J_{neg-sample}(o, v_c, U)=-\log (\sigma(u_o^{\top}v_c))-\Sigma ^K_{k=1}\log(\sigma(-u^{\top}_kv_c))$

$P(w)=U(w)^{\frac 3 4}/Z$

Another way: co-occurrence matrix X (full document and SVD) (LSA, HAL)

* stopwords
  * min(X, t)
  * ignore them all
  * use Pearson correlations instead of counts
* computational cost of SVD ($O(mn^2)$)
* hard to incorporate new words

Combining both: GloVe  $J(\theta)=\frac 1 2 \Sigma^W_{i,j=1}f(P_{ij})(u_i^{\top}v_j-\log P_{ij})^2$

* fast training
* scalable to huge corpora
* good performance even with small corpus and small vectors

$X_{fianl}=U+V$

Word Vector Analogies: $d=\arg\displaystyle\max_i\frac{(x_b-x_a+x_c)^{\top}x_i}{\lVert x_b-x_a+x_c \rVert}$

## Backpropagation

$\frac{\partial}{\partial x}(Wx+b)=W$

$\frac{\partial}{\partial b}(Wx+b)=I$

$\frac{\partial}{\partial u}(u^{\top h})=h^{\top}$

$\frac{\partial}{\partial z}(f(z))=diag(f'(z))$

[downstream gradient] = [upstream gradient] * [local gradient]

Forward: compute result of operation and save intermediate values

Backward: apply chain rule to compute gradient

## Classification

Cross Entropy(one-hot target): $H(p,q)=-\Sigma^C_{c=1}p(c)\log q(c)$

Kullback-Leibler(KL) divergence: $H(p, q)=H(p)+D_{KL}(p\Vert q)$ where $D_{KL}(p\Vert q)=\Sigma^C_{c=1}p(c)\log\frac{p(c)}{q(c)}$

> What happens when we retrain the word vectors?
>
> Those that are in the training data move around and others stay. Retrain the word vector if you have large dataset.

## Overfitting

* Dropout
* Regularization
* Reduce network depth/size
* Reduce input feature dimensionality
* Early stopping
* Max-Norm, Dropconnect, etc.

## Underfitting

* Increase model complexity/size
* Decreasing regularization effects
* Reducing Dropout probability
* Ensemble
* Data Preprocessing
* Batch Normalization
* Curriculum Learning
* Data Augmentation

## Named Entity Recognition

* Person
* Location
* Organizaiton
* None

## Dependency Parsing

Constituency = phrase structure grammar = context-free grammars (CFGs)

* Bilexical affinities
* Dependency distance
* Intervening material
* Valency of heads

Methods:

* Dynamic programming
* Graph algorithms
* Constraint Satisfaction
* Greedy Transition-based parsing

### Shift-reduce parser

(words in buffer, words in stack, set of parsed dependencies, set of actions)

Handling non-projectivity:

* Declare defeat
* Use post-processor
* Add extra transitions
* Use a special parsing mechanism

### Neural Dependency Parsing

embedded vector representations

* Vector representation
* POS tags
* Arc labels

Model: $y=softmax(U\circ ReLU(Wx+b_1)+ b_2)$

## Language Model

> How to learn a language model?
>
> Learn a n-gram Language Model.

$P(x^{(t+1)}\vert x^{(t)},\dots,x^{(1)})=P(x^{(t+1)} \vert x^{(t)},\dots,x^{(t-n+2)})=\frac{P(x^{(t+1)}, x^{(t)},\dots,x^{(t-n+2)})}{P(x^{(t)}, x^{(t-1)},\dots,x^{(t-n+2)})}$

Problems:

* Sparsify: need smoothing and backoff
* Model size: $O(\exp(n))$

Neural Language Model: $y=softmax(U\circ f(We+b_1)+b_2)$

Improvements:

* No sparsity problem
* Model size is $O(n)$

Problems:

* fixed window is too small
* enlarging window enlarges $W$
* window can never be large enough
* do not share weights across the window

RNN Language Model: $y=softmax(U\circ \sigma(W_hh^{(t-1)}+W_ee^{(t)}+b_1)+b_2)$

Evaluation metric: perplexity $PP=\Pi^T_{t=1}(\frac1{\Sigma^{\vert V\vert}_{j=1}y_j^{(t)}\cdot \hat{y}_j^{(t)}})^{1/T}$

## Recurrent Neural Networks (RNN)

Core idea: **Apple the same weights repeatedly**.

Adcantages:

* Can process any length input
* Model size doesn't increase for longer input
* Step $t$ can use information from many steps back
* Weights are shared across timesteps

Disadvantages:

* Slow, hard to parallel
* In practice, difficult to access information from many steps back

Usage:

* part-of-speech tagging
* named entity recognition
* sentiment analysis (take element-wise max or mean of all hidden states are usually better than final hidden state)
* generate text by repeated sampling (speech recoginition, machine translation, summarization)

$\hat{y}^{(t)}=softmax(Uh^{(t)}+b_2)\in \Bbb{R}^{\vert V\vert}$

$h^{(t)}=\sigma(W_hh^{(t-1)}+W_ee^{(t)}+b_1)$

$z^{(t)}=W_hh^{(t-1)}+W_ee^{(t)}+b_1$

$\theta^{(t)}=Uh^{(t)}+b_2$

> What's the derivative $\frac{\partial J^{(t)}}{\partial W_h}$ ? Leave as a chain rule.
>
> Recall $W_h$ appears at every time step. Caculate the sum of gradients w.r.t. each time it appears.

$\frac{\partial h^{(t)}}{\partial h^{(t-1)}}$ can lead to vanishing or exploding gradients.

$\lVert \frac{\partial h_j}{\partial h_{j-1}}\rVert\le\rVert W^T\rVert\lVert diag[f'(h_{j-1})]\rVert\le\beta_W\beta_h$

$\lVert \frac{\partial h_t}{\partial h_k}\rVert=\lVert \Pi^t_{j=k+1}\frac{\partial h_j}{\partial h_{j-1}}\rVert\le(\beta_W\beta_h)^{t-k}$

Gradient problems:

* Backprop in RNNs have a recursive gradient call for hidden layer
* Magnitude of gradients of typical activation functions (sigmoid, relu) lie between 0 and 1. Also depends on repeated multiplicaitons of $W$ matrix
* If gradient magnitude is large/small, increasing timesteps increases/decreases the final magnitude
* RNNs fail to learn long term dependencies

How to solve:

* exploding gradients: gradient clipping(update only when $g\ge threashold$ )
* vanishing gradients: GRUs or LSTMs or Init + ReLUs

> Add L2-norm will help with vanishing gradients?
>
> False. This will put the weights toward 0, which can make it worse.
>
> Add more layers will solve vanishing gradient?
>
> False. This will increase the chance of vanishing gradient problems.

### Gated Recurrent Units (GRU)

$z_t=\sigma(W^{(z)}x_t+U^{(z)}h_{t-1})$

$r_t=\sigma(W^{(r)}x_t+U^{(r)}h_{t-1})$

$\tilde{h}\_{t} = \tanh(Wx_t + r_t \circ Uh_{t-1})$

$h_t=z_t\circ h_{t-1}+(1-z_t)\circ \tilde{h}_t$

Intuition:

* high $r_t$ $\implies$ short-term dependencies
* high $z_t$ $\implies$ long-term dependencies(solves vanishing gradients problem)

> If the update gate $z_t$ is close to 1, the net doesn't update its current state significantly?
>
> True. In this case, $h_t\approx \tilde{h}_t$ .

### Long-Short-Term-Memories (LSTM)

$i_t=\sigma(W^{(i)}x_t+U^{(i)}h_{t-1})$

$f_t=\sigma(W^{(f)}x_t+U^{(f)}h_{t-1})$

$o_t=\sigma(W^{(o)}x_t+U^{(o)}h_{t-1})$

$\tilde{c}\_{t}=\tanh(W^{(c)}x_t+U^{(c)}h_{t-1})$

$c_t=f_t\circ c_{t-1}+i_t\circ \tilde{c}_t$

$h_t=o_t\circ \tanh(c_t)$

Backprop from $c_t$ to $c_{t-1}$ only elementwise multiplication by $f_t$. No longer only depends on $\frac{dh_t}{dh_{t-1}}$.

> The entries of $f_t, i_t, o_t$ are non-negative?
>
> True. The range of sigmoid is (0, 1).

### Bidirectional RNNs

$y_t=g(U[\overrightarrow{h}_t;\overleftarrow{h}_t]+c)$

### Training

* Initialize recurrent matrices to be orthogonal
* Initialize other matrices with a sensible(small) scale
* Initialize forget gate bias to 1: default to remember
* Use adaptive learning rate algorithms: Adam, AdaDelta, ...
* Clip the norm of the gradient
* Either only dropout vertically or learn how to do it right

## Machine Translation

* $P(x\vert y)$ need large amount of parallel data

* $P(x,a\vert y)$ where $a$ is the alignment

* $P(y)$ refers to a language model

Statistical Machine Translation:

* Systems have many separately-designed subcomponents
* Lots of feature engineering
* Require compiling and maintaining extra resources
* Lots of human effort to maintain

Neural Machine Translation (Seq2Seq)

* Encoder RNN produces an encoding of the source sentence and provides inital hidden state for Decoder RNN
* Decoder RNN is a langauge model that generates target sentence conditioned on encoding
* $P(y\vert x)=P(y_1\vert x)P(y_2\vert y1, x)P(y_3\vert y_1,y_2,x)\dots P(y_T\vert y_1,\dots,y_{T-1},x)$
* Use beam search decoding (on each step of decoder, keep track of the k most probable partial translations)

* Better performance (more fluent, better use of context, better use of phrase similarities)
* A single neural network to be optimized end-to-end
* Requires much less human engineering effort
* Less interpretable, hard to debug, difficult to control

Use BLEU(Bilingual Evaluation Understudy) to evaluate: compares machine tranlation to human translation and computes a similarity score based on:

* n-gram precision (usually up to 3 or 4)
* penalty for too short tranlations

Problems:

* out-of-vocabulary words
* domain mismatch
* low-resource language pairs
* maintaining context over longer text
* using common sense is still hard
* NMT picks up biases in training data
* uninterpretable systems do strange things

Improve: use attention

* solves the bottleneck problem
* helps with vanishing gradient problem
* provides some interpretability: alignment for free

Seq2Seq model:

* summarization
* dialogue
* parsing
* code generation

Large-vocab NMT:

* each time train on a smaller vocab $V' \ll V$ 
* test on K most frequent words: unigram prob.

Byte Pair Encoding: most frequent ngram pairs $\to$ a new ngram

Hybrid NMT: mostly at the word level, only go to the character level when needed

## Quasi-Recurrent Neural Network (QRNN)

Take the best and parallelizable parts of RNNs and CNNs.

Parallelism computation across time:

$Z=\tanh(W_z*X)$

$F=\sigma(W_f*X)$

$O=\sigma(W_o*X)$

Element-wise gated recurrence for parallelism across channels:

$h_t=f_t\odot h_{t-1}+(1-f_t)\odot z_t$

## Attention

Attention scores: $e^t=[s_t^Th_1,\dots,s_t^Th_N]\in\Bbb{R}^N$

$\alpha^t=softmax(e^t)\in\Bbb{R}^N$

$a_t=\Sigma^N_{i=1}\alpha^t_ih_i\in\Bbb{R}^h$

Compute $e\in\Bbb{R}^N$ from $h_1,\dots,h_N\in\Bbb{R}^{d_1}$ and $s\in\Bbb{R}^{d_2}$:

* Basic dot-product attention: $e_i=s^{\top}h_i\in\Bbb{R}$
* Multiplicative attention: $e_i=s^{\top}Wh_i\in\Bbb{R}$
* Additive attention: $e_i=v^{\top}\tanh(W_1h_i+W_2s)\in\Bbb{R}$

Applications:

* Pointing to words for language modeling: $p(y_i\lvert x_i)=g\thinspace p_{vocab}(y_i\lvert x_i)+(1-g)p_{ptr}(y_i\lvert x_i)$
* Intra-Decoder attention for summarization
* Machine Translation with Seq2Seq

Encoder attention:

$e_{ti}=f(h_t^d,h_i^e)=h_t^{d^{\top}}W^e_{attn}h_i^e$

$e'\_{ti}=\begin{cases} \exp(e_ti) & \text{if}\space t=1\\\frac{\exp(e_{ti})}{\Sigma_{j=1}^{t-1}\exp(e_{ji})} & \text{otherwise} \end{cases}$

$\alpha^e_{ti}=\frac{e'\_{ti}}{\Sigma^n_{j=1}e'\_{tj}}$

$c_t^e=\Sigma^n_{i=1}\alpha^e_{ti}h^e_i$

Self-attention on decoder:

$e^d_{tt'}=h^{d\top}\_tW^d_{attn}h^d_{t'}$

$\alpha^d_{tt'}=\frac{\exp(e^d_{tt'})}{\Sigma^{t-1}\_{j=1}\exp(e^d_{tj})}$

$c^d_t=\Sigma^{t-1}\_{j=1}\alpha^d_{tj}h^d_j$

Combine softmax and pointers:

$p(u_t=1)=\sigma(W_u[h_t^d\lVert c_t^e\rVert c_t^d]+b_u)$

$p(y_t\lvert u_t=0)=\text{softmax}(W_{out}[h_t^d\lVert c_t^e\rVert c_t^d]+b_{out})$

$p(y_t=x_i\vert u_t=1)=\alpha^e_{ti}$

$p(y_t)=p(u_t=1)p(y_t\vert u_t=1)+p(u_t=0)p(y_t\vert u_t=0)$

### Attention is all you need

$A(q, K, V)=\Sigma_i\frac{e_{q\cdot k_i}}{\Sigma_je^{q\cdot k_j}}v_i$

$A(Q,K,V)=softmax(\frac {QK^{\top}}{\sqrt{d_k}})V$

Self-attention and multi-head attention:

$MultiHead(Q,K,V)=Concat(head_1,\dots,head_h)W^o$

where $head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$

Layer norm:

$\mu^l=\frac1H\Sigma^H_{i=1}a^l_i$

$\sigma^l=\sqrt{\frac1H\Sigma^H_{i=1}(a^l_i-\mu^l)^2}$

$h_i=f(\frac{g_i}{\sigma_i}(a_i-\mu_i)-b_i)$

Added is a positional encoding:

$PE_{pos, 2i}=\sin(pos/10000^{2i/d_{model}})$

$PE_{pos, 2i+1}=\cos(pos/10000^{2i/d_{model}})$

Transformer Decoder: masked decoder self-attention on previously generated outputs.

* byte-pair encodings
* checkpoint averaging
* Adam optimizer with learning rate changes
* Dropout during training at every layer just before adding residual
* label smoothing
* auto-regressive decoding with beam search and length penalties

## Convolutional Neural Networks (CNN)

1d discrete convolution generally: $(f*g)[n]=\Sigma_{m=-M}^Mf[n-m]g[m]$

$x_{1:n}=x_1\oplus x_2\oplus\dots\oplus x_n$

$c_i=f(w^{\top}x_{i:i+h-1}+b)$

$\hat{c}=\max{[c_1, c_2, \dots, c_{n-h+1}]}$

$z=[\hat{c}_1, \dots, \hat{c}_m]$

$y=\text{softmax}(W^{(s)}z+b)$

## Coreference Resolution

Applications:

* Full text understanding
* Machine Translation
* Bialogue Systems

Two steps:

* Detect the montions(easy)
  * Pronouns: POS tagging
  * Named entities: NER system
  * Noun pharses: constituency parser
* Cluster the mentions(hard)

> How to deal with these bad mentions?
>
> Keep all mentions as "candidate mentions".

Coreference Models:

* Mention Pair

  * $J=-\Sigma^N_{i=2}\Sigma^i_{j=1}y_{ij}\log p(m_j, m_i)$, $y_{ij}=1$ if mentions $m_i$ and $m_j$ are coreferent, -1 if otherwise
  * Many mentions only have one clear antecedent, but we want all.
  * Solution: more linguistically plausible

* Mention Ranking

  * Assign each mention its highest scoring candidate antecedent according to the model

  * $J=\sum^N_{i=2}-\log(\sum^{i-1}\_{j=1}\mathbb{1}(y_{ij}=1)p(m_j,m_i))$

  * Non-Neural Coref Model: Features

  * Neural Coref Model

    * Embeddings: previous two words, first word, last word, head word, ... of each mention
    * Distance
    * Document genre
    * Speaker information

  * End-to-End Model

    * Word & character embedding $\to$ BiLSTM $\to$ Attention

    * Do mention detection and coreference end-to-end

      $g_i=[x^{\*}\_{start(i)},x^{\*}\_{end(i)},\hat{x}_i,\phi(i)]$

      $\alpha_t=w_{\alpha}\cdot \text{FFNN}_{\alpha}(x^*_t)$

      $a_{i,t}=\frac{\exp(\alpha_t)}{\sum^{end(i)}_{k=start(i)}\exp(\alpha_k)}$

      $\hat{x}\_i=\sum^{end(i)}\_{t=start(i)}a_{i,t}\cdot x_t$

      $s(i,j)=s_m(i)+s_m(j)+s_a(i,j)$

      $s_m(i)=w_m\cdot \text{FFNN}_m(g_i)$

      $s_a(i,j)=w_a\cdot\text{FFNN}_a([g_i,g_j,g_i\circ g_j,\phi(i,j)])$

* Clustering

  * Current candidate cluster merges depend on previous ones it already made.
  * Metrics: MUC, CEAF, LEA, B-CUBED, BLANC

## Constituency Parsing

Language recursive:

- helpful in disambiguation
- helpful for some tasks to refer to specific phrases
- works better for some tasks to use grammatical tree structure

Recursive neural nets require a tree structure, while recurrent neural nets cannot capture pharses without prefix context and often capture too much of last words in final vector.

### Tree Recursive Neural Network

Input: two candidate children's representations

Outpu: the semantic representation if the two nodes are merged and score of how plausible the new node would be

$score = U^{\top}p$

$p=\tanh(W\begin{bmatrix}c_1 \\\\\\ c_2 \end{bmatrix}+b)$, same $W$ parameters at all nodes of the tree

$score(text, tree)=\sum_{n\in nodes(tree)}s_n$

$J=\sum_is(x_i,y_i)-\max_{y\in A(x_i)}(s(x_i,y)+\triangle(y,y_i))$

$\delta^{(l)}=((W^{(l)})^{\top}\delta^{(l+1)})\circ f'(z^{(l)})$

$\frac{\partial}{\partial W^{(l)}}E_R=\delta^{(l+1)}(a^{(l)})^{\top}+\lambda W^{(l)}$

Differences of backprop in recursion and tree structure:

* sum derivatives of $W$ from all nodes
* split derivatives at each node
* add error messages from parent + node itself

### Syntactically-Untied RNN

Use different composition matrix for different syntactic environments.

Problem: speed.

Solution: compute score only for a subset of trees coming from a simpler, faster model(PCFG).

Compositional Vector Grammar(CVG): PCFG + TreeRNN

### Compositionality Through Recursive Matrix-Vector Spaces

$p=\tanh(W\begin{bmatrix}c_2c_1 \\\\\\ c_1c_2 \end{bmatrix}+b)$

Matrix-Vector RNNs

$p=g(A,B)=W_M\begin{bmatrix}A \\\\\\ B\end{bmatrix}$

> Can an MV-RNN learn how a large syntractic context conveys a semantic relationship?
>
> Build a single compositional semantics for the minimal constituent including both terms.

## Model overview and memory networks

### TreeLSTMs

TreeLSTM = TreeRNN + LSTM

$\tilde{h}\_j=\sum_{k\in C(j)}h_k$

$i_j=\sigma(W^{(i)}x_j+U^{(i)}\tilde{h}_j+b^{(i)})$

$f_{jk}=\sigma(W^{(f)}x_j+U^{(f)}h_k+b^{(f)})$

$o_j=\sigma(W^{(o)}x_j+U^{(o)}\tilde{h}_j+b^{(o)})$

$u_j=\tanh(W^{(u)}x_j+U^{(u)}\tilde{h}_j+b^{(u)})$

$c_j=i_j\odot u_j+\sum_{k\in C(j)}f_{jk}\odot c_k$

$h_j=o_j\odot \tanh(c_j)$

### Neural Architecture Search

* Maintain the controller (RNN)
* Sample architecture A with probability $p$
* Train a child network with architecture A to get accuracy R
* Compute gradient of $p$ and scale it by R to update the controller

### Dynamic Memory Network

* Input module

  Standard GRU or BiGRU

* Question module

  $q_t=GRU(v_t, q_{t-1})$

* Episodic Memory module

  $h_i^t=g_i^tGRU(s_i,h^t_{i-1})+(1-g^t_i)h^t_{i-1}$, last hidden state $m^t$

  gates are activated if sentence relevant to the question or memory.

  $z_i^t=[s_i\circ q;s_i\circ m^{t-1};\lvert s_i-q\rvert;\lvert s_i-m^{t-1}\rvert]$

  $Z^t_i=W^{(2)}\tanh(W^{(1)}z^t_i+b^{(1)})+b^{(2)}$

  $g^t_i=\frac{\exp(Z^t_i)}{\sum^{M_i}_{k=1}\exp(Z^t_k)}$

* Answer Module

  $a_t=GRU([y_{t-1},q],a_{t-1})$

  $y_t=softmax(W^{(a)}a_t)$

Related work: Neural Turing Machine.



## Semi-Supervised Learning

* Pre-training
  * first train an unsupervised model on unlabeled data, then train it on the labeled data
  * Word2Vec (skip-gram, CBOW, GloVe, etc.)
  * Auto-Encoder
  * Strategies:
    * CoVe
    * ELMo
* Self-training
  * train the model on the labeled data, then use the model to label the unlabeled data
  * Online self-training: $J(\theta)=CE(y_i,p(y\lvert x_i,\theta))+CE(onehot(argmax(p(y\lvert x_j,\theta))),p(y\lvert x_j,\theta))$
  * hard targets work better than soft targets
* Consistency regularization
  * $J(\theta)=CE(p(y\lvert x_j,\theta),p(y\lvert x_j+\eta,\theta))$ where $\eta$ is a vector with a random direction and a small magnitude $\epsilon$
  * Apply to NLP:
    * Add noise to the word embedding(noise should be chosen adversarially)
      * Compute the gradient of the loss with respect to the input, then add epsilon times the gradient to the input.
      * $\eta=\epsilon\frac{\nabla_xJ}{\lVert\nabla_xJ\rVert}$
    * Word dropout
      * randomly(10%-20%) replace words in the input with a special REMOVED token: $J(\theta)=CE(p(y\lvert x_j,\theta),p(y\lvert dropwords(x_j), \theta))$
    * Cross-view Consistency
      * train the model across many different views of the input at once
      * instead of running full the model multiple times, add multiple "auxiliary" softmax layers to the model
      * $J(\theta)=\Sigma_{i=1}^kCE(p(y\lvert x_j,\theta),p_{view_i}(y\lvert x_j,\theta))$ 
      * forward and backward auxiliary softmax layer, attention dropout, etc.

## Next

3 equivalent NLP-Complete Super Tasks

* Language Model
* Question Answering
* Dialogue System

Limits for deep NLP:

* Comprehensive QA
* Multitask learning
* Combined multimodel, logical and memory-based reasoning
* Learning from few examples
