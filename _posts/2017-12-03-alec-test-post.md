---
layout:     post
title:      k-server problem
date:       2017-12-03 11:21:29
summary:    Recent breakthrough of the k-server problem on HST
categories: 
---

The $$k$$-server problem is one of the most important and most well-studied problems in the field of online algorithms. The goal of this blog is to describe the problem to you and give you some favor what are the underlying technique used in the recent breakthrough by [Sebastien Bubeck, Michael Cohen, James Lee, Aleksander Madry and me](https://arxiv.org/abs/1711.01085). For a detailed blog with proofs, please see [here](https://blogs.princeton.edu/imabandit/2017/12/16/k-server-part-1-online-learning-and-online-algorithms/).

## Background

In this problem, we control the movement of a set of $$k$$ servers on a fixed metric space of $$n$$ vertices. In each iteration, we are
given a request $$r$$. If there is no server at that location, we must choose a server, move it to that location and pay the movement cost of this server. Our goal is to minimize the total distances all servers move. This problem is very general and is originally proposed to model problems related to cache management. We call an algorithm is $$f(k,n)$$-competitive if for any request sequence, its total movement is at most $$f(k,n)\cdot\text{OPT}$$ where $$\text{OPT}$$ is the optimal movement cost if the whole request sequence is given in the beginning. A priori, it is unclear that any competitive algorithm exists because we are comparing against an algorithm that knows the whole request sequence and is free to make any moves.

To illustrate the learning aspect, consider a $$2$$-server problem on a graph with 3 vertices $$\{a_{1},a_{2},b\}$$. Suppose that $$d(a_{1},a_{2})=1$$, $$d(a_{i},b)=2017$$ for $$i=1,2$$ and the request is 

$$
a_{1},b,a_{2},b,a_{1},b,a_{2},b,\cdots\cdots.
$$

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
x^{(k+1)}=\mathsf{argmin}\_{x\in K}\eta\cdot v^{(k)\top}x^{(k)}+D\_{\Phi}(x;x^{(k)})
\\]
where $$\eta$$ is the step size, $$\Phi$$ is some convex functions on $$K$$ called mirror map and the Bregman divergence associated to $$\Phi$$
defined by
\\[
D_{\Phi}(y;x):=\Phi(y)-\Phi(x)-\nabla\Phi(x)^{\top}(y-x).
\\]
When $$x$$ is very close to $$y$$, $$D_{\Phi}(y;x)\sim(y-x)^{\top}\nabla^{2}\Phi(x)(y-x)$$ and hence mirror descent is simply moving $$x$$ towards $$-v$$ direction while making sure the point is in $$K$$ and it is not too far from the previous point in $$\nabla^{2}\Phi(x)$$ norm.

For me, a general wisdom, when faced a new learning problem, is to check if mirror descent or some of its variant is good. See my favorite example [here](https://arxiv.org/abs/1607.03084).

## Weighted Complete Graph

Unfortunately, applying mirror descent to the $$k$$-server problem is not as easy as picking a good mirror map as I wished. Let me first describe the algorithm for the complete graphs with the metric of the form $$d(i,j)=\omega_{i}+\omega_{j}$$. For this and many other graphs, it is known how to turn a fractional solution $$x(t)\in\mathbb{R}^{n}$$ that is feasible ($$0\leq x_{i}(t)\leq1$$ and $$\sum x_{i}(t)=k$$) to an integral solution $$\widetilde{x}\in\{0,1\}^{n}$$ with the movement cost bounded by $$O(1)\cdot\int_{0}^{T}\omega_{i}\left|\frac{dx_{i}}{dt}\right|dt$$ (a natural continuous definition of movement cost). Therefore, it
suffices to proposed a fractional algorithm (the first prerequisite for applying mirror descent).

Our algorithm is motivated by the $$O(\log k)$$ competitive-algorithm by [Bansal, Buchbinder and Naor in 2007](http://www.win.tue.nl/~nikhil/pubs/pot-wt2.pdf). To make its relation to mirror descent clear, I describe our process
here as a discrete process with an infinitesimally small step size $$\eta$$. Instead of working on the fractional server $$x(t)$$, our algorithm is defined on the fractional missing server $$y(t):=1-x(t)$$:

* Let $$K=\{0\leq y\leq1,\sum y_{i}=n-k\}$$.
* Let $$\Phi(y)=\sum_{i=1}^{n}\omega_{i}(y_{i}+\frac{1}{2k})\log(y_{i}+\frac{1}{2k})$$.
* When the request for the vertex $$\ell$$ arrives
* - While $$y_{\ell}^{(k+1)}>0$$.
* - + $$y^{(k+1)}=\mathsf{argmin}_{y\in K}\eta\cdot e_{\ell}^{\top}y+D_{\Phi}(y;y^{(k)})$$ where $$e_{\ell}$$ is the coordinate vector at $$\ell$$.

In short, when the request arrives at $$\ell$$, we run the mirror descent with the cost $$e_{\ell}$$ until all anti-mass $$y$$ leave the coordinate $$\ell$$.

The reason of using $$e_{\ell}$$ is to put a "cost" at coordinate $$\ell$$ that forces every anti-mass at $$y_{\ell}$$ leaves (equivalently, attracting fractional servers to move to $$\ell$$). Since $$K$$ is a simplex, the standard choice of the mirror map is $$\sum_{i}\omega_{i}y_{i}\log y_{i}$$. However, this mirror map is not suitable because its gradient $$(1+\log y_{i})_{i}$$
blows up on the boundary. Following the idea in BBN07, we shift all variables by $$\Theta(\frac{1}{k})$$ in the mirror map.

Since the step size is infinitesimally small, 
\\[
D_{\Phi}(y;y^{(k)}) = (y-y^{(k)})^{\top}\nabla^{2}\Phi(y^{(k)})(y-y^{(k)}) =\sum_{i}\frac{\omega_{i}}{y_{i}^{(k)}+\frac{1}{2k}}(y-y^{(k)})^{2}.
\\]
Using this, one can show that the algorithm is moving the anti-mass $$y$$ from coordinate $$\ell$$ to coordinate $$j$$ with a rate proportionally to $$\frac{y_{i}^{(k)}+\frac{1}{2k}}{\omega_{i}}$$. Namely, the algorithm tends to move the server from vertices with smaller weight and less fractional server mass to the request. One can show this algorithm has a $$O(\log k)$$ competitive ratio.

## Hierarchically Separated Tree

For the $$k$$-server problem, it suffices to solve on an HST [BBNM11](https://arxiv.org/abs/1110.1580). Given a tree $$T=(V,E)$$ with vertex weights $$\omega>0$$, we define the metric on the leaf $$\mathcal{L}$$ by $$d_{\omega}(\ell,\ell')=\omega_{\text{lca}(\ell,\ell')}$$ where $$\text{lca}(\ell,\ell')$$ is the least common ancestor of $$\ell$$ and $$\ell'$$. We call this tree is a HST if $$\omega_{v}\leq\omega_{u}/6$$ whenever $$v$$ is a child of $$u$$ and we call the metric space $$(\mathcal{L},d_{\omega})$$ is a HST metric.

There is two main question to be decided first.
1. How do we represent the solution?
2. What is the mirror map we use?

One natural choose to represent anti-mass $$y$$ is to use
\\[
B:=\\{y\in[0,1]^{V}:\ y_{r}=n-k\text{ and }y_{u}=y_{v}\text{ whenever }v\text{ is a child of }u\\}
\\]
where $$r$$ is the root of the tree. The main issue of this representation is that it cannot distinguish between the case we need exactly 1 server in that subtree or we need 2 servers with 50% probability. Another small issue is that some constraint is never active. Using the fact that the algorithm always moving servers to the request in one dimension until $$y_{\ell}=0$$, one can show that $$1\geq y\geq0$$ and $$y_{u}\geq y_{v}$$ are never active.

Fixing these issues, we have a new representation

\\[
K:=\\{y:\ y_{i,r}=1\_{i>k}, \\\\ \qquad \qquad \qquad \qquad \sum_{i\leq\left|S\right|}x\_{u,i}\leq\sum\_{(v,j)\in S}x_{v,j} \  \forall S  \\}.
\\]
where $$S$$ are sets of pairs $$(v,j)$$ with $$v$$ is a child of $$u$$.
The first constraint indicates there are only $$k$$ servers in total and the second constraint indicates that the number of servers in
the children of a vertex $$u$$ is at most the number of server at $$u$$.

The mirror map we pick is simply the generalization of the mirror map on simplex
\\[
\Phi(x):=\sum_{u\in V}\omega_{u}\sum_{i\geq1}(y_{u,i}+\delta)\log(y_{u,i}+\delta)
\\]
where the shift $$\delta=\frac{1}{2017k}$$.

Except for some technical issue, our algorithm for HST is same as the algorithm described for the complete graph except using this new
$$K$$ and mirror map $$\Phi$$.

## Open Problems

The proof of the classical mirror descent is clean, short and optimal. Unfortunately, our proof is slightly longer (i.e. few pages) and the bound we get for HST does not sound optimal. We hope to get the ultimate algorithm for $$k$$-server at least for HST, however, our algorithm seems not the one from the BOOK. So, what is the algorithm from the BOOK for the $$k$$-server problem for HST?

On the other hand, it is interesting to see if HST is necessary for getting an $$\log^{O(1)}k$$-competitive algorithm for general graph. This problem is even open for a path. Or HST is the right tool and that I am too naive on this?
