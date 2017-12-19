---
layout:     post
title:      k-server problem
tlink:		http://www.google.com
date:       2017-12-03 11:21:29
summary:    Recent breakthrough of the k-server problem on HST
categories: 
---

The $$k$$-server problem is one of the most important and most well-studied problems in the field of online algorithms. The goal of this blog is to describe the problem to you and give you some favor what are the underlying technique in the recent breakthrough by Sebastien Bubeck, Michael Cohen, James Lee, Aleksander Madry and me. For a more detailed blog with proofs, please see [here](https://blogs.princeton.edu/imabandit/2017/12/16/k-server-part-1-online-learning-and-online-algorithms/).

# Background

In this problem, we control the movement of a set of $$k$$ servers on a fixed metric space of $$n$$ vertices. In each iteration, we are
given a request $$r$$ and if there is no server at that location, we must choose a server, move it to that location and pay the movement cost this server. Our goal is to minimize the total distances all servers move. This problem is very general and is originally proposed to model problems related to cache management. We call an algorithm is $$f(k,n)$$-competitive if for any request sequence, its total movement is at most $$f(k,n)\cdot\text{OPT}$$ where $$\text{OPT}$$ is the optimal movement cost if the whole request sequence is given in the beginning. A priori it is unclear that any competitive algorithm exists because we are comparing against an algorithm that knows the whole request sequence and is free to make any moves.

To illustrate the learning aspect, consider a $$2$$ servers problem on a graph with 3 vertices $$\{a_{1},a_{2},b\}$$. Suppose that $$d(a_{1},a_{2})=1$$, $$d(a_{i},b)=2017$$ for $$i=1,2$$ and the request is 
\\[
a_{1},b,a_{2},b,a_{1},b,a_{2},b,\cdots\cdots.
\\]
Since $$a_{i}$$ and $$b$$ are so far away, the optimal strategy for this sequence is to put a server on $$a_{i}$$ and a server on $$b$$. However, if $$b$$ appears less frequent than once per 2017 iterations, a better strategy would put both servers on $a_{i}$ most of the time. Therefore, any memoryless algorithm is not competitive and that we indeed need to learn something. 

One main focus for the $$k$$-server problem is to achieve competitive ratio independent of $$n$$ because the case $$k\ll n$$ is common in many settings. Two of the major open problems is to find a $k$-competitive deterministic algorithm and a $$O(\log k)$$-competitive randomized algorithm for the $$k$$-server problem. For the deterministic problem, [Koutsoupias and Papadimitriou](https://doi.org/10.1145/210118.210128) gave a deterministic algorithm achieves a competitive ratio of $$2k-1$$ on any metric space. However, there is not too many progress on the randomize problem. In particular, there is no any $$o(k)$$ competitive algorithm except for very basic classes of graphs such as weighted complete graphs. Due to its importance and the gap between the lower and upper bound, the following "easier" problem had been one major target of the field. And in my opinion, this was the most important conjecture on online algorithms.

---
(Weak randomized $$k$$-server conjecture) There is a randomized algorithm for the $$k$$-server problem on any graph with competitive ratio $$\log^{O(1)}(k)$$.
---

The previous best competitive ratio is $$O(\log^{3}(n)\log^{2}(k))$$ (by [Bansal, Buchbinder, Naor and Madry in 2011](https://arxiv.org/abs/1110.1580v1). In our paper, we give an algorithm with competitive ratio $$O(\log^{2}(k))$$ on hierarchically separated trees (HST, a much larger classes of graphs) and $$O(\log^{2}(k)\log(n))$$ on general graph. Soon after our paper,
James Lee, our coauthor, developed upon our paper and gave an algorithm with competitive ratio $$O(\log^{6}(k))$$, finally resolving the weak randomized $$k$$-server conjecture!
