---
title: Basic NLP Knowledge
categories: [Note, Deep Learning, NLP]
---

[_Notes of Stanford NLP course_](http://web.stanford.edu/~jurafsky/NLPCourseraSlides.html)

<!-- more -->

## Preference

### Language Technology

* Mostly solved:
    * Spam detection
    * Part-of-speech (POS) tagging
    * Named Entity Recognition (NER)
* Making good progress
    * Sentiment analysis
    * Co-reference resolution
    * Word sense disambiguation (WSD)
    * Parsing
    * Machine translation (MT)
    * Information Extraction (IE)
* Still really hard
    * Question answering (QA)
    * Paraphrase
    * Summarization
    * Dialog

### Why NLP difficult?

* Non-standard Language
* segmentation issues
* idioms
* neologisms
* world knowledge
* tricky entity names

### Basic skills

* Regular Expressions
* Tokenization
* Word Normalization and Stemming
* Classifier
    * Decision Tree
    * Logistic Regression
    * SVM
    * Neural Nets

### Edit Distance

* Used for
    * Spell correction
    * Computational Biology
* Basic operations
    * Insertion
    * Deletion
    * Substitution
* Algorithm
    * Levenshtein
    * Back trace
    * Needleman-Wunsch
    * Smith-Waterman

## Language Model

### Probabilistic Language Models

* Machine translation
* Spell correction
* Speech Recognition
* Summarization
* Question answering

### Markov Assumption

$$ P(\omega_1 \omega_2 \dots \omega_n) \approx  \prod_i P(\omega_i | \omega_{i-k} \dots \omega_{i-1}) $$

### Unigram Model

$$ P(\omega_1 \omega_2 \dots \omega_n) \approx  \prod_i P(\omega_i) $$

### Bigram Model

$$ P(\omega_i | \omega_1 \omega_2 \dots \omega_{i-1}) \approx  \prod_i P(\omega_i | \omega{i-1}) $$

### Add-k Smoothing

$$ P_{Add-k}(\omega_i|\omega{i-1})=\tfrac{c(\omega_{i-1},\omega_i)+k}{c(\omega_{i-1})+kV} $$

### Unigram prior smoothing

$$ P_{Add-k}(\omega_i|\omega_{i-1})=\tfrac{c(\omega_{i-1},\omega_i)+m(\tfrac{1}{V})}{c(\omega_{i-1})+m} $$

### Smoothing Algorithm

* Good-Turing
* Kneser-Ney
* Witten-Bell

### Spelling Correction

* tasks:
    * Spelling error detection
    * Spelling error correction
        * Autocorrect
        * Suggest a correction
        * Suggestion lists
* Real word spelling errors:
    * For each word $w$, generate candidate set
    * Choose best candidate
    * Find the correct word $w$
* Candidate generation
    * words with similar spelling
    * words with similar pronunciation
* Factors that could influence `p(misspelling|word)`
    * source letter
    * target letter
    * surrounding letter
    * the position in the word
    * nearby keys on the keyboard
    * homology on the keyboard
    * pronunciation
    * likely morpheme transformations

## Text Classification

### Used for:

* Assigning subject categories, topics, or genres
* Spam detection
* Authorship identification
* Age/gender identification
* Language identification
* Sentiment analysis
* ...

### Methods: Supervised Machine Learning

* Naive Bayes
* Logistic Regression
* Support Vector Machines
* k-Nearset Neighbors

### Naive Bayes

$$ C_{MAP}=arg\max_{c\in C}P(x_1,x_2,\dots,x_n|c)P(c) $$

* Laplace (add-1) Smoothing
* Used for Spam Filtering
* Training data:
    * No training data: manually written rules
    * Very little data:
        * Use Naive Bayes
        * Get more labeled data
        * Try semi-supervised training methods
    * A reasonable amount of data:
        * All clever Classifiers:
            * SVM
            * Regularized Logistic Regression
        * User-interpretable decision trees
    * A huge amount of data:
        * At a cost
            * SVM (train time)
            * kNN (test time)
            * Regularized Logistic Regression can be somewhat better
        * Naive Bayes
* Tweak performance
    * Domain-specific
    * Collapse terms
    * Upweighting some words

### F Measure

Precision: % of selected items that are correct

Recall: % of correct items that are selected

/ | correct | not correct
--|---|--
 selected | tp | fp
 not selected | fn | tn

$$ F=\tfrac{1}{\alpha \tfrac{1}{P} +(1-\alpha)\tfrac{1}{R}}=\tfrac{(\beta^2+1)PR}{\beta^2P+R} $$

### Sentiment Analysis

* Typology of Affective States
    * Emotion
    * Mood
    * Interpersonal stances
    * Attitudes (Sentiment Analysis)
    * Personality traits
* Baseline Algorithm
    * Tokenization
    * Feature Extraction
    * Classification
        * Naive Bayes
        * MaxEnt (better)
        * SVM (better)
* Issues
    * HTML, XML and other markups
    * Capitalization
    * Phone numbers, dates
    * Emoticons

### Sentiment Lexicons

* semi-supervised learning of lexicons
    * use a small amount of information
    * to bootstrap a lexicon
    * adjectives conjoined by "and" have same polarity
    * adjectives conjoined by "but" do not
* Process
    * label seed set of adjectives
    * expand seed set to conjoined adjectives
    * supervised classifier assigns "polarity similarity" to each word pair
    * clustering for partitioning
* Turney Algorithm
    * extract a phrasal lexicon from reviews
    * learn polarity of each phrase
    * rate a review by the average polarity of its phrases
* Advantages
    * domain-specific
    * more robust
* Assume classes have equal frequencies:
    * if not balanced: need to use F-scores
    * severe imbalancing also can degrade classifier performance
    * solutions:
        * Resampling in training
        * Cost-sensitive learning
            * penalize SVM more for misclassification of the rare thing
* Features:
    * Negation is important
    * Using all words works well for some tasks (NB)
    * Finding subsets of words may help in other tasks

## Features

### Joint and Discriminative

* Joint (generative) models: place probabilities over both observed data and the hidden stuff ---- P(c,d)
    * N-gram models
    * Naive Bayes classifiers
    * Hidden Markov models
    * Probabilistic context-free grammars
    * IBM machine translation alignment models
* Discriminative (conditional) models: take the data as given, and put a probability over hidden structure given the data ---- `P(c|d)`
    * Logistic regression
    * Conditional loglinear
    * Maximum Entropy models
    * Conditional Random Fields
    * SVMs
    * Perceptron

### Features

* Feature Expectations:
    * Empirical count
    * Model expectation
* Feature-Based Models:
    * Text Categorization
    * Word-Sense Disambiguation
    * POS Tagging

### Maximum Entropy

$$ \log P(C|D,\lambda)=\sum_{(c,d)\in (C,D)}\log P(c|d,\lambda)=\sum_{(c,d)\in(C,D)}\log \tfrac{exp \sum_{i} \lambda_if_i(c,d)}{\sum_{c'} exp\sum_i \lambda_if_i(c',d)} $$

* Find the optimal parameters
    * Gradient descent (GD), Stochastic gradient descent (SGD)
    * Iterative proportional fitting methods: Generalized Iterative Scaling (GIS) and Improved Iterative Scaling (IIS)
    * Conjugate gradient (CG), perhaps with preconditioning
    * Quasi-Newton methods - limited memory variable metric (LMVM) methods, in particular, L-BFGS
* Feature Overlap
    * Maxent models handle overlapping features well
    * Unlike NB, there is no double counting
* Feature Interaction
    * Maxent models handle overlapping features well, but do not automatically model feature interactions
* Feature Interaction
    * If you want to interaction terms, you have to add them
    * A disjunctive feature would also have done it
* Smoothing:
    * Issues of scale
        * Lots of features
        * Lots of sparsity
        * Optimization problems
    * Methods
        * Early stopping
        * Priors (MAP)
        * Regularization
        * Virtual Data
        * Count Cutoffs

### Named Entity Recognition (NER)

* The uses:
    * Named entities can be indexed, linked off, etc.
    * Sentiment can be attributed to companies or products
    * A lot of IE relations are associations between named entities
    * For question answering, answers are often named entities
* Training:
    * Collect a set of representative training documents
    * Label each token for its entity class or other
    * Design feature extractors appropriate to the text and classes
    * Train a sequence classifier to predict the labels form the data
* Inference
    * Greedy
        * Fast, no extra memory requirements
        * Very easy to implement
        * With rich features including observations to the right, it may perform quite well
        * Greedy, we make commit errors we cannot recover from
    * Beam
        * Fast, beam sizes of 3-5 are almost as good as exact inference in many cases
        * Easy to implement
        * Inexact: the globally best sequence can fall off the beam
    * Viterbi
        * Exact: the global best sequence is returned
        * Harder to implement long-distance state-state interactions
* CRFs
    * Training is slower, but CRFs avoid causal-competition biases
    * In practice usually work much the same as MEMMs

### Relation Extraction

* How to build relation extractors
    * Hand-written patterns
        * High-precision and low-recall
        * Specific domains
        * A lot of work
    * Supervised machine learning
        * MaxEnt, Naive Bayes, SVM
        * Can get high accuracies with enough training data
        * Labeling a large training set is expensive
        * Brittle, don't generalize well to different genres
    * Semi-supervised and unsupervised
        * Bootstrapping (using seeds)
            * Find sentences with these pairs
            * Look at the context between or around the pair and generalize the context to create patterns
            * Use the patterns for grep for more pairs
        * Distance supervision
            * Doesn't require iteratively expanding patterns
            * Uses very large amounts of unlabeled data
            * Not sensitive to genre issues in training corpus
        * Unsupervised learning from the web
            * Use parsed data to train a "trustworthy tuple" classifier
            * Single-pass extract all relations between NPs, keep if trustworthy
            * Assessor ranks relations based on text redundancy

### POS Tagging

* Performance
    * About 97% currently
    * But baseline is already 90%
    * Partly easy because
        * Many words are unambiguous
        * You get points for them and for punctuation marks
    * Difficulty
        * ambiguous words
        * common words can be ambiguous
* Source of information
    * knowledge of neighboring words
    * knowledge of word probabilities
    * word
    * lowercased word
    * prefixes
    * suffixes
    * capitalization
    * word shapes
* Summary
    * the change from generative to discriminative model does not by itself result in great improvement
    * the higher accuracy of discriminative models comes at the price of much slower training

## Parsing

* Treebank
    * reusability of the labor
        * many parser, POS taggers, etc.
        * valuable resource for linguistics
    * broad coverage
    * frequencies and distributional information
    * a way to evaluate systems
* Statistical parsing applications
    * high precision question answering
    * improving biological named entity finding
    * syntactically based sentence compression
    * extracting interaction in computer games
    * helping linguists find data
    * source sentence analysis for machine translation
    * relation extraction systems
* Phrase structure grammars (= context-free grammars, CFGs) in NLP
    * G = (T, C, N, S, L, R)
        * T is a set of terminal symbols
        * C is a set of preterminal symbols
        * N is a set of nonterminal symbols
        * S is the start symbol
        * L is the lexicon, a set of items of the form X -> x
        * R is the grammar, a set of items of the form X -> $\gamma$
        * e is the empty symbol

### Probabilistic Parsing

* Probabilistic - or stochastic - context-free grammars (PCFGs)
    * G = (T, N, S, R, P)
        * P is a probability function
    * Chomsky Normal Form
        * Reconstructing n-aries is easy
        * Reconstructing unaries/empties is trickier
        * Binarization is crucial for cubic time CFG parsing
    * Cocke-Kasami-Younger (CKY) Constituency Parsing
        * Unaries can by incorporated into the algorithm
        * Empties can be incorporated
        * Binarization is vital
    * Performance
        * Robust
        * Partial solution for grammar ambiguity
        * Give a probabilistic language model
        * The problem seems to be that PCFGs lack the lexicalization of a trigram model

### Lexicalized Parsing

* Charniak
    * Probabilistic conditioning is "top-down" like a regular PCFG, but actual parsing is bottom-up, somewhat like the CKY algorithm we saw
* Non-Independence
    * The independence assumptions of a PCFG are often too strong
    * We can relax independence assumptions by encoding dependencies into the PCFG symbols, by state splitting (sparseness)
* Accurate Unlexicalized Parsing
    * Grammar rules are not systematically specified to the level of lexical items
    * Closed vs. open class words
* Learning Latent Annotations
    * brackets are known
    * base categories are known
    * induce subcategories
    * clever split/merge category refinement
    * EM, like Forward-Backward for HMMs, but constrained by tree

### Dependency Parsing

* Methods
    * Dynamic programming (like in the CKY algorithm)
    * Graph algorithm
    * Constraint Satisfaction
    * Deterministic Parsing
* Sources of information
    * Bilexical affinities
    * Dependency distance
    * Intervening material
    * Valency of heads
* MaltParser
    * Greedy
    * Bottom up
    * Has
        * a stack
        * a buffer
        * a set of dependency arcs
        * a set of actions
    * Each action is predicted by a discriminative classifier (SVM)
    * No search
    * Provides close to state of the art parsing performance
    * Provides very fast linear time parsing
* Projective
    * Dependencies from a CFG tree using heads, must be projective
    * But dependency theory normally does allow non-projective structure to account for displaced constituents
    * The arc-eager algorithm only builds projective dependency trees
* Stanford Dependencies
    * Projective
    * Can be generated by postprocessing headed phrase structure parses, or dependency parsers like MaltParser or the Easy-First Parser

## Information Retrieval

* Used for
    * web search
    * e-mail search
    * searching your laptop
    * corporate knowledge bases
    * legal information retrieval

### Classic search

* User task
* Info need
* Query & Collection
* Search engine
* Results

### Initial stages of text processing

* Tokenization
* Normalization
* Stemming
* Stop words

### Query processing

* AND
    * "merge" algorithm
* Phrase queries
    * Biword indexes
        * false positives
        * bigger dictionary
    * Positional indexes
        * Extract inverted index entries for each distinct term
        * Merge their doc:position lists to enumerate all positions
        * Same general method for proximity searches
    * A positional index is 2-4 as large as a non-positional index
    * Caveat: all of this holds for "English-like" language
    * These two approaches can be profitably combined

### Ranked Retrieval

* Advantage
    * Free text queries
    * large result sets are not an issue
* Query-document matching scores
    * Jaccard coefficient
        * Doesn't consider term frequency
        * Length normalization needed
    * Bag of words model
        * Term frequency (tf)
        * Log-frequency weighting
        * Inverse document frequency (idf)

### tf-idf weighting

$$ W_{t,d}=(1+\log tf_{t,d})\times \log_{10}(N/df_t) $$

* Best known weighting scheme in information retrieval

### Distance: cosine(query, document)

$$ \cos(\vec q,\vec d)=\tfrac{\vec q \bullet \vec d}{|\vec q||\vec d|}=\tfrac{\vec q}{|\vec q|}\bullet \tfrac{\vec d}{|\vec d|}=\tfrac{\sum^{|V|}_{i=1}q_id_i}{\sqrt{\sum^{|V|}_{i=1}q_i^2}\sqrt{\sum^{|V|}_{i=1}d^2_i}} $$

### Weighting

* Many search engines allow for different weightings for queries vs. documents
* A very standard weighting scheme is: Inc.Itc
* Document: logarithmic tf, no idf and cosine normalization
* Query: logarithmic tf idf, cosine normalization

### Evaluation

* Mean average precision (MAP)

## Semantic

### Situation

* Reminder: lemma and wordform
* Homonymy
    * Homographs
    * Homophones
* Polysemy
* Synonyms
* Antonyms
* Hyponymy and Hypernymy
* Hyponyms and Instances

### Applications of Thesauri and Ontologies

* Information Extraction
* Information Retrieval
* Question Answering
* Bioinformatics and Medical Informatics
* Machine Translation

### Word Similarity

* Synonymy and similarity
* Similarity algorithm
    * Thesaurus-based algorithm
    * Distributional algorithms

### Thesaurus-based similarity

$LCS(c_1,c_2)=$ The most informative (lowest) node in the hierarchy subsuming both $c_1$ and $c_2$

$$ Sim_{path}(c_1,c_2)=\tfrac{1}{pathlen(c_1,c_2)} $$

$$ Sim_{resnik}(c_1,c_2)=-\log P(LCS(c_1,c_2)) $$

$$ Sim_{lin}(c_1,c_2)=\tfrac{1\log P(LCS(c_1,c_2))}{\log P(c_1)+\log P(c_2)} $$

$$ Sim_{jiangconrath}(c_1,c_2)=\tfrac{1}{\log P(c_1)+\log P(c_2)-2\log P(LCS(c_1,c_2))} $$

$$ Sim_{eLesk}(c_1,c_2)=\sum_{r,q\in RELS}overlap(gloss(r(c_1)),gloss(q(c_2))) $$

* Evaluating
    * Intrinsic
        * Correlation between algorithm and human word similarity ratings
    * Extrinsic (task_based, end-to-end)
        * Malapropism (Spelling error) detection
        * WSD
        * Essay grading
        * Taking TOEFL multiple-choice vocabulary tests
* Problems
    * We don't have a thesaurus for every language
    * recall
        * missing words
        * missing phrases
        * missing connections between senses
        * works less well for verbs, adj.

### Distributional models of meaning

* For the term-document matrix: tf-idf
* For the term-context matrix: Positive Pointwise Mutual Information (PPMI) is common

$$ PMI(w_1,w_2)=\log_2\tfrac{P(w_1,w_2)}{P(w_1)P(w_2)} $$

* PMI is biased toward infrequent events
    * various weighting schemes
    * add-one smoothing

### Question Answering

* Question processing
    * Detect question type, answer type (NER), focus, relations
    * Formulate queries to send a search engine
* Passage Retrieval
    * Retrieval ranked documents
    * Break into suitable passages and re-rank
* Answer processing
    * Extract candidate answers
    * Rank candidates

#### Approaches

* Knowledge-based
    * build a semantic representation of the query
    * Map from this semantics to query structured data or resources
* Hybrid
    * build a shallow semantic representation of the query
    * generate answer candidate using IR methods
    * Score each candidate using richer knowledge sources

#### Answer type taxonomy

* 6 coarse classes
    * Abbreviation
    * Entity
    * Description
    * Human
    * Location
    * Numeric
* 50 finer classes
* Detection
    * Hand-written rules
    * Machine Learning
    * Hybrids
* Features
    * Question words and phrases
    * Part-of-speech tags
    * Parse features
    * Named Entities
    * Semantically related words

#### Keyword selection

1. Non-stop words
2. NNP words in recognized named entities
3. Complex nominals with their adjectival modifiers
4. Other complex nominals
5. Nouns with their adjectival modifiers
6. Other nouns
7. Verbs
8. Adverbs
9. QFW word (skipped in all previous steps)
10. Other words

#### Passage Retrieval

* IR engine retrieves documents using query terms
* Segment the documents into shorter units
* Passage ranking
    * number of named entities of the right type in passage
    * number of query words in passage
    * number of question N-grams also in passage
    * proximity of query keywords to each other in passage
    * longest sequence of question words
    * rank of the document containing passage

#### Features for ranking candidate answers

* answer type match
* pattern match
* question keywords
* keyword distance
* novelty factor
* apposition features
* punctuation location
* sequences of question terms

#### Common Evaluation Metrics

* Accuracy
* Mean Reciprocal Rank (MRR)

$$ MRR = \tfrac{\sum_{i=1}^N \tfrac{1}{rank_i}}{N} $$

### Summarization

* Applications
    * outlines or abstracts
    * summaries
    * action items
    * simplifying
* Three stages
    * content selection
    * information ordering
    * sentence realization
* salient words
    * tf-idf
    * topic signature
        * mutual information
        * log-likelihood ratio (LLR)

$$
weight(w_i)=
    \begin{cases}
        1,& if -2\log \lambda(w_i)>10 \\ 0,& otherwise
    \end{cases}
$$

* Supervised content selection problem
    * hard to get labeled training data
    * alignment difficult
    * performance not better than unsupervised algorithm
* ROUGE (Recall Oriented Understudy for Gisting Evaluation)
    * Intrinsic metric for automatically evaluating summaries
        * based on BLEU
        * not as good as human evaluation
        * much more convenient

$$ ROUGE-2=\tfrac{\sum_{x\in \{RefSummaries\}}\sum_{bigrams\:i\in S}\min(count(i,X),count(i,S))}{\sum_{x\in\{RefSummaries\}}\sum_{bigrams\:i\in S}count(i,S)} $$

* Maximal Marginal Relevance (MMR)
    * Iteratively (greedily)
        * Relevant: high cosine similarity to the query
        * Novel: low cosine similarity to the summary
    * Stop when desired length
* Information Ordering
    * Chronological ordering
    * Coherence
    * Topical ordering
