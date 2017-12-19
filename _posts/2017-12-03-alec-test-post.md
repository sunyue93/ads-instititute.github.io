---
layout:     post
title:      k-server problem
tlink:		http://www.google.com
date:       2017-12-03 11:21:29
summary:    Recent breakthrough of the k-server problem on HST
categories: 
---

The $$k$$-server problem is one of the most important and most well-studied problems in the field of online algorithms. The goal of this blog is to describe the problem to you and give you some favor what are the underlying technique used in the recent breakthrough by [Sebastien Bubeck, Michael Cohen, James Lee, Aleksander Madry and me](https://arxiv.org/abs/1711.01085). For a detailed blog with proofs, please see [here](https://blogs.princeton.edu/imabandit/2017/12/16/k-server-part-1-online-learning-and-online-algorithms/).

## Background

In this problem, we control the movement of a set of $$k$$ servers on a fixed metric space of $$n$$ vertices. In each iteration, we are
given a request $$r$$. If there is no server at that location, we must choose a server, move it to that location and pay the movement cost of this server. Our goal is to minimize the total distances all servers move. This problem is very general and is originally proposed to model problems related to cache management. We call an algorithm is $$f(k,n)$$-competitive if for any request sequence, its total movement is at most $$f(k,n)\cdot\text{OPT}$$ where $$\text{OPT}$$ is the optimal movement cost if the whole request sequence is given in the beginning. A priori, it is unclear that any competitive algorithm exists because we are comparing against an algorithm that knows the whole request sequence and is free to make any moves.

To illustrate the learning aspect, consider a $$2$$-server problem on a graph with 3 vertices $$\{a_{1},a_{2},b\}$$. Suppose that $$d(a_{1},a_{2})=1$$, $$d(a_{i},b)=2017$$ for $$i=1,2$$ and the request is 
\\[
a_{1},b,a_{2},b,a_{1},b,a_{2},b,\cdots\cdots.
\\]
Since $$a_{i}$$ and $$b$$ are so far away, the optimal strategy for this sequence is to put a server on $$a_{i}$$ and a server on $$b$$. However, if $$b$$ appears less frequent than once per $$2017$$ iterations, a better strategy would put both servers on $$a_{i}$$ most of the time. Therefore, any memoryless algorithm is not competitive and that we indeed need to learn something. 

One main focus for the $$k$$-server problem is to achieve competitive ratio independent of $$n$$ because of the common case $$k\ll n$$. Two of the major open problems is to find (a $$k$$-competitive deterministic algorithm)[https://en.wikipedia.org/wiki/K-server_problem] and a $$O(\log k)$$-competitive randomized algorithm for the $$k$$-server problem. For the deterministic problem, [Koutsoupias and Papadimitriou](https://doi.org/10.1145/210118.210128) gave a deterministic algorithm achieves a competitive ratio of $$2k-1$$ on any metric space. However, there is not too many progress on the randomize problem. In particular, there is no any $$o(k)$$ competitive algorithm except for very basic classes of graphs such as weighted complete graphs. Due to its importance and the gap between the lower and upper bound, the following "easier" problem had been a major target of the field. And in my opinion, this was the most important conjecture in online algorithms.

> **(Weak randomized $$k$$-server conjecture)** There is a randomized algorithm for the $$k$$-server problem on any graph with competitive ratio $$\log^{O(1)}(k)$$.

The previous best competitive ratio is $$O(\log^{3}(n)\log^{2}(k))$$ (by [Bansal, Buchbinder, Naor and Madry in 2011](https://arxiv.org/abs/1110.1580v1)). In our paper, we give an algorithm with competitive ratio $$O(\log^{2}(k))$$ on hierarchically separated trees (HST, a much larger classes of graphs) and $$O(\log^{2}(k)\log(n))$$ on general graph. Soon after our paper,
James Lee, our coauthor, developed upon our paper and gave an algorithm with competitive ratio $$O(\log^{6}(k))$$, finally resolving the weak randomized $$k$$-server conjecture!


## Online Learning and Mirror Descent

Our algorithm is based on mirror descent with a multiscale entropy. So, let me describe an online learning problem and the mirror descent algorithm for it. 

In this problem, there is a fixed given convex set $$K$$. In the $$k^{th}$$ iteration, we select a vector $$x^{(k)}\in K$$, then the adversary select a vector $$v^{(k)}$$ on $$K$$ and we receive a loss $$v^{(k)\top}x^{(k)}$$ for that iteration. Our goal is to minimize the regret (the difference between your loss and the loss of the optimal fixed strategy)
\\[
\sum_{k=1}^{T}v^{(k)\top}x^{(k)}-\min_{x\in K}\sum_{k=1}^{T}v^{(k)\top}x.
\\]
This problem can be solved by mirror descent: 
\\[
x^{(k+1)}=\text{argmin}_{x\in K}\eta\cdot v^{(k)\top}x^{(k)}-+D_{\Phi}(x;x^{(k)})
\\]
where $$\eta$$ is the step size, $$\Phi$$ is some convex functions on $$K$$ called mirror map and the Bregman divergence associated to $$\Phi$$
defined by
\\[
D_{\Phi}(y;x):=\Phi(y)-\Phi(x)-\nabla\Phi(x)^{\top}(y-x).
\\]
When $$x$$ is very close to $$y$$, $$D_{\Phi}(y;x)\sim(y-x)^{\top}\nabla^{2}\Phi(x)(y-x)$$ and hence mirror descent is simply moving $$x$$ towards $$-v$$ direction while making sure the point is in $$K$$ and it is not too far from the previous point in $$\nabla^{2}\Phi(x)$$ norm.

For me, a general wisdom, when faced a new learning problem, is to check if mirror descent or some of its variant is good. See my favorite example [here](https://arxiv.org/abs/1607.03084).

