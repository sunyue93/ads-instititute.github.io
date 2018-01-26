---
layout:     post
title:      Proximal point algorithm revisited, episode 2. The prox-linear algorithm
date:       2018-01-24 17:02:00 -0700
summary:    Revisiting the proximal point method. Composite models and the prox-linear algorithm.
author:     Dmitriy Drusvyatskiy
image:      images/prox.png
image_url:  http://sites.math.washington.edu/~ddrusv/
categories: blog
---

This is episode 2 of the three-part series that revisits the classical proximal
point algorithm. See the [first post on this topic](../proximal-point) for 
introduction and notation.

<a name="sec2"></a>The prox-linear algorithm
=========================

For well-structured weakly convex problems, one can hope for faster
numerical methods than the subgradient scheme. In this episode, I will
focus on the composite problem class $$\mathcal{C}$$.
To simplify the exposition, I will assume $$L=1$$, which can always be
arranged by rescaling.

Since composite functions are weakly convex, one could apply the
proximal point method directly, while setting the parameter
$$\nu\leq\beta^{-1}$$. Even though the proximal subproblems are strongly
convex, they are not in a form that is most amenable to convex
optimization techniques. Indeed, most convex optimization algorithms are
designed for minimizing a sum of a convex function and a composition of
a convex function with a *linear* map. This observation suggests
introducing the following modification to the proximal-point algorithm.
Given a current iterate $$x_t$$, the *prox-linear method* sets

$$
\begin{aligned}
x_{t+1}=\underset{x}{\operatorname{argmin}} \{F(x;x_t)+\tfrac{\beta}{2}\|x-x_t\|^2\},
\end{aligned}
$$

where $$F(x;y)$$ is the local convex model

$$
F(x;y):=g(x)+h\left(c(y)+\nabla c(y)(x-y)\right).
$$ 

In other words,
each proximal subproblem is approximated by linearizing the smooth map
$$c$$ at the current iterate $$x_t$$.

The main advantage is that each subproblem is now a sum of a strongly
convex function and a composition of a Lipschitz convex function with a
linear map. A variety of methods utilizing this structure can be
formally applied; e.g. smoothing (Nesterov 2005), saddle-point
(Nemirovski 2004; Chambolle and Pock 2011), and interior point
algorithms (Nesterov and Nemirovskii 1994; Wright 1997). Which of these
methods is practical depends on the specifics of the problem, such as
the size and the cost of vector-matrix multiplications.

It is instructive to note that in the simplest setting of additive
composite problems
(Example 1), the prox-linear method reduces to the
popular proximal-gradient algorithm or ISTA (Beck and Teboulle 2012).
For nonlinear least squares, the prox-linear method is a close variant
of Gauss-Newton.

Recall that the step-size of the proximal point method provides a
convenient stopping criteria, since it directly relates to the gradient
of the Moreau envelope -- a smooth approximation of the objective
function. Is there such an interpretation for the prox-linear method?
This question is central, since termination criteria is not only used to
stop the method but also to judge its efficiency and to compare against
competing methods.

The answer is yes. Even though one can not evaluate the gradient
$$\|\nabla F_{\frac{1}{2\beta}}\|$$ directly, the scaled step-size of the
prox-linear method 

$$
\mathcal{G}(x):=\beta(x_{t+1}-x_t)
$$ 

is a good
surrogate (Drusvyatskiy and Paquette 2016 Theorem 4.5):

$$
\tfrac{1}{4} \|\nabla F_{\frac{1}{2\beta}}(x)\| \leq \|\mathcal{G}(x)\|\leq 3\|\nabla F_{\frac{1}{2\beta}}(x)\|.
$$

In particular, the prox-linear method will find a point $$x$$ satisfying
$$\|\nabla F_{\frac{1}{2\beta}}(x)\|^2\leq\varepsilon$$ after at most
$$O\left(\frac{\beta(F(x_0)-\inf F)}{\varepsilon}\right)$$ iterations. In
the simplest setting when $$g=0$$ and $$h(t)=t$$, this rate reduces to the
well-known convergence guarantee of gradient descent, which is black-box
optimal for $$C^1$$-smooth nonconvex optimization (Carmon et al. 2017b).

It is worthwhile to note that a number of improvements to the basic
prox-linear method were recently proposed. Cartis,
Gould, and Toint (2011) discuss trust region variants and their
complexity guarantees, while Duchi and Ruan (2017b) propose stochastic
extensions of the scheme and prove almost sure convergence.
Drusvyatskiy and Paquette (2016) discuss overall complexity guarantees
when the convex subproblems can only be solved by first-order methods,
and proposes an inertial variant of the scheme whose convergence
guarantees automatically adapt to the near-convexity of the problem.

Local rapid convergence
-----------------------

Under typical regularity conditions, the prox-linear method exhibits the
same types of rapid convergence guarantees as the proximal point method.
I will illustrate with two intuitive and widely used regularity
conditions, yielding local linear and quadratic convergence,
respectively.

 A local minimizer $$\bar x$$ of $$F$$ is *$$\alpha$$-tilt-stable* if there
exists $$r>0$$ such that the solution map

$$
M: v\mapsto \underset{x\in B_r(\bar x)}{\operatorname{argmin}} \left\{ F(x)-\langle v,x \rangle\right\}
$$

is $$1/\alpha$$-Lipschitz around $$0$$ with $$M(0)=\bar x$$.

This condition might seem unfamiliar to convex optimization specialist.
Though not obvious, tilt-stability is equivalent to a uniform quadratic
growth property and a subtle localization of strong convexity of $$F$$.
See Drusvyatskiy and Lewis (2013) or Drusvyatskiy, Mordukhovich, and
Nghia (2014) for more details on these equivalences. Under the
tilt-stability assumption, the prox-linear method initialized
sufficiently close to $$\bar x$$ produces iterates that converge at a
linear rate $$1-\alpha/\beta$$.

The second regularity condition models sharp growth of the function
around the minimizer. Let $$S$$ be the set of all stationary points of
$$F$$, meaning $$x$$ lies in $$S$$ if and only if the directional derivative
$$F'(x;v)$$ is nonnegative in every direction $$v\in {\mathbb R}^d$$.

 A local minimizer $$\bar x$$ of $$F$$ is *sharp* if there exists $$\alpha>0$$
and a neighborhood $$\mathcal{X}$$ of $$\bar x$$ such that

$$
F(x)\geq  F({\rm proj}_S(x))+c\cdot {\rm dist}(x,S)\qquad\forall x\in \mathcal{X}.
$$

Under the sharpness condition, the prox-linear method initialized
sufficiently close to $$\bar x$$ produces iterates that converge
quadratically.

For well-structured problems, one can hope to justify the two regularity
conditions above under statistical assumptions. The recent work of Duchi
and Ruan (2017a) on the phase retrieval problem is an
interesting recent example. Under mild statistical assumptions on the
data generating mechanism, sharpness is assured with high probability.
Therefore the prox-linear method (and even subgradient methods (Davis,
Drusvyatskiy, and Paquette 2017)) converge rapidly, when initialized
within a constant relative distance of an optimal solution.


References
==========
See [here](../proximal-point#references).

