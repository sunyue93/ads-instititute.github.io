---
layout:     post
title:      Proximal point algorithm revisited, episode 1: the proximally guided subgradient method
date:       2018-01-24 17:01:00 -0700
summary:    Revisiting the proximal point method: the proximally guided subgradient method for stochastic optimization.
author:     Dmitriy Drusvyatskiy
image:      images/prox.png
image_url:  http://sites.math.washington.edu/~ddrusv/
categories: blog
---

This is episode 1 of the three part series that revisits the classical proximal
point algorithm. See the [previous post](2018-01-24-proximal-point.md) for 
introduction and notation.

<a name="sec1"></a>The proximally guided subgradient method
========================================

As the first example of contemporary applications of the proximal point
method, consider the problem of minimizing the expectation:[^3]

$$
\min_{x\in {\mathbb R}^d}~ F(x)=\mathbb{E}_{\zeta} f(x,\zeta).
$$ 

Here,
$$\zeta$$ is a random variable, and the only access to $$F$$ is by sampling
$$\zeta$$. It is difficult to overstate the importance of this problem
class (often called *stochastic approximation*) in large-scale
optimization; see e.g. Bottou and Bousquet (2008); Bartlett, Jordan, and
McAuliffe (2006).

When the problem is convex, the stochastic subgradient method (Polyak
and Juditsky 1992; Robbins and Monro 1951; Nemirovski et al. 2008) has
strong theoretical guarantees and is often the method of choice. In
contrast, when applied to nonsmooth and nonconvex problems, the behavior
of the method is poorly understood. The recent paper (Davis and Grimmer
2017) shows how to use the proximal point method to guide the
subgradient iterates in this broader setting, with rigorous guarantees.

Henceforth, assume that the function $$x\mapsto f(x,\zeta)$$ is
$$\rho$$-weakly convex and $$L$$-Lipschitz for each $$\zeta$$. Davis and
Grimmer (2017) proposed the scheme outlined below.

#### Proximally guided stochastic subgradient method

-   **Data**: $$x_0\in {\mathbb R}^d$$, $$\{j_t\}\subset\mathbb{N}$$,
    $$\{\alpha_j\}\subset{\mathbb R}_{++}$$
-   **For** $$t=0,\ldots,T$$ **do**
    -   Set $$y_0=x_t$$
    -   **For** $$j=0,\ldots,j_t-2$$ **do**
        - Sample $$\zeta$$ and choose $$v_j\in\partial (f(\cdot,\zeta)+\rho\|\cdot-x_t\|^2)(y_j)$$
        - Set $$y_{j+1}= y_j-\alpha_jv_j$$
    -   Set $$x_{t+1}= \frac{1}{j_t}\sum_{j=0}^{j_t-1}y_j$$

The method proceeds by applying a proximal point method with each
subproblem approximately solved by a stochastic subgradient method. The
intuition is that each proximal subproblem is $$\rho/2$$-strongly convex
and therefore according to well-known results (e.g. Lacoste-Julien,
Schmidt, and Bach (2012); Rakhlin, Shamir, and Sridharan (2012); Hazan and
Kale (2011); Juditsky and Nesterov (2014)), the stochastic subgradient
method should converge at the rate $$O(\frac{1}{T})$$ on the subproblem,
in expectation. This intuition is not quite correct because the
objective function of the subproblem is not globally Lipschitz -- a key
assumption for the $$O(\frac{1}{T})$$ rate. Nonetheless, the authors show
that warm-starting the subgradient method for each proximal subproblem
with the current proximal iterate corrects this issue, yielding a
favorable guarantees (Davis and Grimmer 2017 Theorem 1).

To describe the rate of convergence, set
$$j_t=t+\lceil 648\log(648)\rceil$$ and $$\alpha_j=\tfrac{2}{\rho(j+49)}$$
in the Proximally guided stochastic subgradient method. Then the scheme
will generate an iterate $$x$$ satisfying

$$
\mathbb{E}_{\zeta}[\|\nabla F_{2\rho}(x)\|^2]\leq \varepsilon
$$ 

after
at most

$$
O\left(\frac{\rho^2(F(x_0)-\inf  F)^2}{\varepsilon^2}+\frac{L^4 \log^{4}(\varepsilon^{-1})}{\varepsilon^2}\right)
$$

subgradient evaluations. This rate agrees with analogous guarantees for
stochastic gradient methods for smooth nonconvex functions (Ghadimi and
Lan 2013). It is also worth noting that convex constraints on $$x$$ can be
easily incorporated into the Proximally guided stochastic subgradient
method by introducing a nearest-point projection in the definition of
$$y_{j+1}$$.


[^3]: For simplicity of the exposition, the minimization problem is
    unconstrained. Simple constraints can be accommodated using a
    projection operation.


References
==========
See [here](2018-01-24-proximal-point.md#references).
