---
layout: default
title: Modeling Non-Linear Time Series With Particle Filtering Part II (Importance Sampling)
---

This post is primarily about importance sampling, which plays an important role (sorry about the pun) in the particle filter. There is an insane amount of resources on the subject of Monte Carlo methods. I, by no means, intend to give a full description. My goal is to describe a couple of basic ideas I've picked up over the past year and relate them to the previous post. My view of Monte Carlo methods is strictly limited to their use in Bayesian statistics, but they can be used in many other places.

I view Monte Carlo methods as excellent tools for evaluating complex integrals. For example, suppose we are interested in integrating

$$
\int_{\mathbb{R}} \ln(x^2) \frac{1}{\sqrt{2\pi}}\exp\left\{-\frac{x^2}{2} \right\} dx .
$$

This is $$E\left\{ln(X^2)\right\}$$, where $$X \sim \mathcal{N}(0,1)$$. This seems challenging enough (I just made this up so hopefully it is sufficiently "too challenging" to serve as a motivating example). To approximate this integral, the Monte Carlo algorithm is as follows:

1. First,
      * sample $$X^{(n)} \sim \mathcal{N}(0,1)$$, for $$n = 1,\ldots, N$$
2. Then, 
      * compute $$\frac{1}{N}\sum_{n=1}^{N} \ln\{(x^{(n)})^2\}$$


We can compute this integral in Python:

```python
# We will use scipy and numpy a lot
import scipy.stats as ss
import numpy as np
# Set N to be a large value
N = 5000000
# Generate random normals for scipy
X = ss.norm.rvs(0,1,N)

# Monte Carlo estimator
np.sum(np.log(X**2)) / N
```
    -1.2695391283999222
    
Voila! we have 

$$
\int_{\mathbb{R}} \ln(x^2) \frac{1}{\sqrt{2\pi}}\exp\left\{-\frac{x^2}{2} \right\} dx \approx -1.27
$$. 

So why does this work? By the Strong Law of Large Numbers:

$$
\lim_{N \rightarrow \infty}\frac{1}{N}\sum_{n=1}^{N} h(X^{(i)}) \overset{a.s.}{\rightarrow} E\{h(X^{(n)})\} \} = \int_{\mathbb{R}}h(x) f(x) dx ,
$$

where the above is "almost sure" convergence (similar to pointwise convergence but in a probabilistic context) for $$X^{(n)} \overset{i.i.d.}{\sim} f(x)$$. Of course, we need $$E[h(X)] < \infty$$ too. To provide more evidence, let's compute an integral we know the answer to.

If you're familiar with the history of Bayesian statistics, you might know that Thomas Bayes showed

$$
\int_{0}^{1} {n\choose x}p^{x}(1-p)^{n-x} dp = \int_{0}^{1} {n\choose x}p^{x}(1-p)^{n-x} \frac{(1)^{1}}{0!}p^{1-1}(1-p)^{1-1} = \frac{1}{n+1} .
$$

Let's just say $$n = 10$$ and $$x = 5$$. The answer should be $$\frac{1}{10 + 1}$$.  


```python
from scipy.special import comb
# We will use matplot too!
import matplotlib.pyplot as plt

# Generate beta random variables (Unif(0,1),here)
P = ss.beta.rvs(a=1,b=1,size=500000)
# Calculate absolute error from 1/11
abs(252 * (np.sum( (P**5) * (1-P)**(5)) / 500000) - (1/11))
```
    2.2093377856413632e-05

Monte Carlo integration works beautifully for the particle filter (as we will see). Why? Because the error is independent of the dimension! For example, consider the variance of the simple Monte Carlo estimator:

$$
V\left\{\frac{1}{N}\sum_{n=1}^{N} h(X^{(n)})\right\} = \frac{1}{N}V\left\{h(X^{(1)})\right\} .
$$

We will assume $$ V\left\{h(X^{(1)})\right\} < \infty $$. Furthermore, by the Central Limit Theorem

$$
\frac{1}{N}\sum_{n=1}^{N} h(X^{(n)}\overset{d}{\rightarrow} \mathcal{N}\left(E\{h(X)\},\frac{V\left\{(h(X^{(1)})\right\}}{N}\right) ,
$$

as $$N \rightarrow \infty$$. So the point is that our estimator is approximating the correct thing with an error of $$\frac{V\left\{(h(X^{(1)})\right\}}{N}$$. Want to reduce your error by a decimal place? Then take ten times the number of current samples!


Now suppose we cannot simulate $$X^{(n)} \sim f(x)$$ (so $$f(x)$$ is some crazy density function). We have a serious problem, right? Fortunately, we learned to multiply and divide as children:

$$E\left\{h(X)\right\} = \int_{\mathbb{R}} h(x) f(x) dx = \int_{\mathbb{R}} h(x) f(x) \frac{q(x)}{q(x)} dx$$ .

Wait, so what if we just simulate from something easier (like $$X^{(i)} \sim q(x)$$, maybe?) We can just tweak our algorithm:

1. First,
      * sample $$X^{(n)} \sim q(x)$$, for $$n = 1,\ldots, N$$.
2. Then, 
      * compute $$\frac{1}{N}\sum_{n=1}^{N} \frac{h(x^{(n)})f(x^{(i)}}{q(x^{(i)})}$$ .
    
Since we know

$$
\begin{aligned}
\frac{1}{N}\sum_{n=1}^{N} \frac{h(x^{(n)})f(x^{(n)})}{q(x^{(n)})} &\rightarrow E\left\{\frac{h(x^{(n)})f(x^{(n)})}{q(x^{(n)})} \right\} \\
&= \int_{\mathbb{R}} h(x) f(x) \frac{q(x)}{q(x)} dx \\
&= \int_{\mathbb{R}} h(x) f(x) dx \\
&= E\left\{h(X)\right\} ,
\end{aligned}
$$

where the first expectation is with respect to $$q(x)$$ and the final expectation is with respect to $$f(x)$$. This trick is called importance sampling (and it stems from a simple algebraic manipulation!)

The following ratio is the _importance weight_ : 

$$
w_{t}^{(n)} = \frac{f(X^{(n)})}{q(X^{(n)})} .
$$

We want the variance of the importance weights to be small. For example, if the proposal $$q(x)$$ _actually is_ the target $$f(x)$$, then the ratio is one (constant) and the variance is zero!

There are a couple of requirements regarding $$q(x)$$:

1. The support of $$q(x)$$ must be equal to the support of $$f(x)$$. For example, if $$f(x)$$ is a Gaussian distribution then $$q(x)$$ cannot be a beta distribution, since we would only generate values between $$(0,1)$$ (which can never _truly_ approximate a Gaussian!)

2. For all intents and purposes, the tails of $$q(x)$$ must be "thicker" than the tails of $$f(x)$$.

I don't recall seeing (2) as a formal proof, but I'll give a motivating example. Note that importance sampling is not necessary here since we can simulate from the target (so this is just for illustration). Suppose our target is a standard Cauchy and the proposal is a standard normal:

$$
\begin{aligned}
\pi(x) &\sim \mathcal{C}(0,1) \\
q(x) &\sim \mathcal{N}(0,1) .
\end{aligned}
$$

What is the variance of a weight for a draw $$X^{(i)} \sim q(x)$$? Well, let's calculate the second moment:

$$
\begin{aligned}
E\left(\frac{\pi(X^{(i)})}{q(X^{(i)})}\right)^2 &= \int_{\mathbb{R}} \frac{(\pi(x^{(i)}))^{2}}{(q(x^{(i)}))^{2}}q(x^{(i)})dx^{(i)} \\
&= \int_{\mathbb{R}} \frac{\pi(x^{(i)})}{q(x^{(i)})} \pi(x^{(i)}) dx^{(i)} \\
&= \int_{\mathbb{R}} \frac{\sqrt{2 \pi}}{ \pi^2} \frac{e^{\frac{(x^{(i)})^2}{2}}}{(1+(x^{(i)})^2)^2} dx^{(i)} \\
&> \int_{1}^{\infty} \frac{\sqrt{2 \pi}}{ \pi^2} \frac{1 + \frac{(x^{(i)})^2}{2} + \frac{(x^{(i)})^4}{4(2)} + \frac{(x^{(i)})^6}{8(6)}}{1+2(x^{(i)})^2 + (x^{(i)})^4} dx^{(i)} \\
&> \int_{1}^{\infty} \frac{\sqrt{2 \pi}}{ \pi^2} \frac{(x^{(i)})^6}{8(6)(4)(x^{(i)})^4} dx^{(i)} = \infty .
\end{aligned}
$$

I illustrate this in Python below (I use a Monte Carlo estimator to compute $$\int_{\mathbb{R}} \frac{\pi(x^{(i)})}{q(x^{(i)})} \pi(x^{(i)}) dx^{(i)}$$). The intuitive argument is as follows: if we propose from a Cauchy distribution, it is not unlikely to observe some huge value. However, such a huge value is unlikely under the Gaussian distribution. In this case, $$q(x) \approx 0$$ and we have a division by zero problem in the importance ratio, making the weight infinite. Notice that if we reverse the roles of the distributions, then the weights are finite (since the tails of the Cauchy distribution dominate the tails of the normal). 


```python
# Set size
# Note that sometimes computational costs should be accounted
# for when setting N. I set it large here cause there's no reason
# not to!
N = 5000000
# Generate random Cauchy random variables
X = ss.cauchy.rvs(0,1,N)
# Calculate Monte Carlo estimate
# Hopefully that's large enough to convince you...
print(1/N * np.sum((ss.cauchy.pdf(X,0,1) / ss.norm.pdf(X,0,1)),dtype=np.int64))

# Now let's switch roles!
# Generate random normal random variables
X = ss.cauchy.rvs(0,1,N)
# Calculate Monte Carlo estimate
# Much better!
print(1/N * np.sum((ss.norm.pdf(X,0,1) / ss.cauchy.pdf(X,0,1)),dtype=np.int64))
```
    1545253656203.8818
    0.6849063999999999
    
```python
# Create grid of points to evaluate PDF
x_grid = np.linspace(-10,10,5000)

# Creates plot below using scipy functions
# Notice that the tails of the Cauchy PDF dominate those of the normal!
plt.figure(figsize=(10,8))
plt.plot(x_grid,ss.cauchy.pdf(x_grid,0,1))
plt.plot(x_grid, ss.norm.pdf(x_grid,0,1))
plt.legend(['Cauchy PDF', 'Normal PDF'])
```
![Image description](/images/output_6_1.png)

There's one possible complication with importance sampling. If we cannot sample from $$f(x)$$, there's usually a good reason why. One common problem in Bayesian statistics is the normalization constant in the posterior distribution.

As an example, suppose we have a normal model with a Cauchy prior: 

$$
\begin{aligned}
X \sim f(X | \theta) = \mathcal{N}(\theta, 1) \\
\pi(\theta) \sim \mathcal{C}(0,1) .
\end{aligned}
$$

The posterior distribution follows from Bayes' rule:

$$
\pi(\theta | x) = \frac{f(x | \theta) \pi(\theta)}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} .
$$

What!? How do you compute 

$$\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta$$ 

analytically? This is going to be a problem in our importance sampler estimator, even if $$\theta \sim q(\theta)$$ is easy to sample from:

$$
\frac{1}{N} \frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta}  \sum_{n=1}^{N} \frac{h(\theta^{(n)})f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} .
$$

This renders importance sampling useless and concludes this post...


just joking. Fortunately, we know how to divide. Recall

$$
w^{(n)} = \frac{f(x | \theta) \pi(\theta)}{q(\theta)} \frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} .
$$

Now replace $$N$$ with $$\sum_{n=1}^{N}w^{(n)}$$. Our new estimator is

$$
\frac{1}{\sum w^{(n)}} \sum_{n=1}^{N} h(\theta^{(n)})w^{(n)} .
$$

Why in the world would we do such a thing? Let's be explicit

$$
\begin{aligned}
\frac{1}{\sum w^{(n)}} \sum_{n=1}^{N} h(\theta^{(n)})w^{(n)} &= \frac{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta}}{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta}\sum \frac{f(x |\theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})}} \sum_{n=1}^{N} h(\theta^{(n)})\frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} \\
&= \frac{1}{\sum \frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})}} \sum_{n=1}^{N} h(\theta^{(n)})\frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} .
\end{aligned}
$$


Notice what happened? The normalization constant disappeared! This is called the _self-normalized importance sampler_. So all we need in these situations is two things:

1. Be able to sample from $$q(\theta)$$.

2. Be able to _evaluate_ $$q(\theta), f(\theta), \pi(\theta)$$, and $$h(\theta)$$

The bad news? The estimator is biased for any fixed $$n$$. The good news? Who really cares about unbiasedness! (it _is_ asymptotically unbiased). Why are these things true? Let's look at the bottom piece of 

$$
\frac{1}{\sum \frac{f(x |\theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})}} \sum_{n=1}^{N} h(\theta^{(n)})\frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} = \frac{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} \frac{1}{N}}{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} \frac{1}{N}\sum \frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})}} \sum_{n=1}^{N} h(\theta^{(n)})\frac{f(x |\theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} .
$$

So we sample $$\theta^{(n)} \sim q(\theta)$$. Then we have 

$$
\begin{aligned}
\frac{1}{N}\sum_{n=1}^{N} \frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{\left(\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta \right) q(\theta^{(n)})} \overset{a.s.}{\rightarrow} E\left\{\frac{f(x | \theta) \pi(\theta)}{\left(\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta \right) q(\theta)}  \right\} &= \int_{\mathbb{R}} \frac{f(x | \theta) \pi(\theta)}{\left(\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta \right) q(\theta)} q(\theta) \\
&= \int_{\mathbb{R}} \frac{f(x | \theta) \pi(\theta)}{\left(\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta \right)} \\
&= 1
\end{aligned}
$$

Okay, so the denominator tends to one almost surely. We know that the numerator tends to the thing we want (the only thing we have changed this far is the denominator):

$$
\frac{1}{N}\sum_{n=1}^{N} h(\theta^{(n)}) \frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{\left(\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta \right) q(\theta^{(n)})} \overset{a.s.}{\rightarrow} E\left\{h(\theta)\right\}
$$

Putting these two results together

$$
\frac{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} \frac{1}{N}}{\frac{1}{\int_{\mathbb{R}}f(x | \theta) \pi(\theta) d\theta} \frac{1}{N}\sum \frac{f(x | \theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})}} \sum_{n=1}^{N} h(\theta^{(n)})\frac{f(x |\theta^{(n)}) \pi(\theta^{(n)})}{q(\theta^{(n)})} \overset{a.s.}{\rightarrow} \frac{ E\left\{h(\theta) \right\} }{(1)} = E\left\{h(\theta)\right\} .
$$

The above doesn't seem obviously true. It is, though, nonetheless. This is a little off-topic, but here's a simple proof I wanted to do for practice. Those without an introductory measure theory background can skip.  

Consider a probability space $$(\Omega,\mathcal{B},\lambda)$$. Suppose $$X_{n}(\omega) \overset{a.s.}{\rightarrow} Z(\omega)$$ and $$Y_{n}(\omega) \overset{a.s.}{\rightarrow} c $$, for $$c \in \mathbb{R}$$. Then by the Continuous Mapping Theorem, $$1/Y(\omega) \overset{a.s.}{\rightarrow} 1/c$$. Let 

$$
\begin{aligned}
A &= \{\omega: \lim_{n \rightarrow \infty} X_{n}(\omega) = Z(\omega) \} \\
B &= \left\{\omega: \lim_{n \rightarrow \infty} \frac{1}{Y_{n}(\omega)} = \frac{1}{c} \right\} \\
C &= \left\{\omega: \lim_{n \rightarrow \infty} \frac{X_{n}(\omega)}{Y_{n}(\omega)} = \frac{Z(\omega)}{c} \right\} .
\end{aligned}
$$

Consider an $$\omega \in A \cap B$$. For this $$\omega$$, $$\lim\limits_{n \rightarrow \infty}\frac{X_{n}(\omega)}{Y_{n}(\omega)} \rightarrow \frac{Z(\omega)}{c}$$ (this is simply a fact regarding limits from real analysis, provided $$Y_{n}(\omega) \neq 0$$). Thus, $$\omega \in C$$. Hence, $$A \cap B \subset C$$ and it follows $$P(A \cap B) \leq P(C)$$. But $$P(A \cap B) = 1$$! If this isn't clear, notice that since $$P(A) = P(B) = 1$$ by assumption,

$$
\begin{aligned}
P(A \cup B) &= P(A) + P(B) - P(A \cap B) \\
&= 2 - P(A \cap B) .
\end{aligned}
$$

Thus, for this to make sense, $$P(A \cap B) = 1$$. It follows that $$P(C) = 1$$ as well and we are done. There are a few technical pieces regarding the case of $$Y_{n}(\omega) = 0$$, but the gist is that the probability of this occurring is zero (this is what differentiates almost sure convergence from pointwise convergence).

Okay, sorry about that. I have used that fact many times without proving it hah! Anyways, the self-normalized importance sampler is (asymptotically) unbiased. I'll end this post by relating these facts to the previous post.

Recall that at time $$t$$ we are interested in drawing samples from 

$$
\pi(x_{0:t} | y_{0:t}) = \frac{g(y_{t} | x_{t})\pi(x_{0:t}| y_{0:t-1})}{\pi(y_{t} | y_{0:t-1})} ,
$$

where the denominator is a component from the normalization constant

$$ 
Z_{t} = \int_{X_{0:t}}\pi(x_{0:t},y_{0:t}) dx_{0:t} = \pi(y_{0:t}) = \pi(y_{t} | y_{0:t-1})\pi(y_{0:t-1}) .
$$

In other words, we cannot practically compute 

$$\pi(y_{t} | y_{0:t-1})$$ 

Consequently, we cannot sample from the target distribution either. That's fine, we can use the self-normalized importance sampler!

Suppose we want to estimate the average value of the state at time $$t$$ . In this case, 

$$h(X_{0:t}) = X_{0:t}$$ 

(remember, this is a $$t+1$$ - dimensional quantity). Then we assign an importance distribution 

$$X_{0:t} \sim q(x_{0:t} | y_{0:t})$$ 

(it does not have to depend on $$y_{0:t}$$ , but it can). Perform the following:

1. First,

      *  Sample from the importance distribution 
      
      $$X_{0:t} \sim q(x_{0:t} | y_{0:t})$$ 
      
2. Then

      * Compute the normalization term 
      
      $$\sum_{n=1}^{N} \frac{g(y_{t} | x^{(n)}_{t})\pi(x^{(n)}_{0:t}| y_{0:t-1})}{q(x^{(n)}_{0:t} | y_{0:t})} = \sum_{n=1}^{N} w^{(n)}_{0:t}$$

3. Finally.
 
      * Compute the estimate 
      
      $$\sum_{i=1}^{N} \frac{1}{\sum_{n=1}^{N} w^{(n)}_{0:t}} h(x_{0:t}^{(i)}) \frac{g(y_{t} | x^{(n)}_{t})\pi(x^{(n)}_{0:t}| y_{0:t-1})}{q(x^{(n)}_{0:t} | y_{0:t})} = \sum_{i=1}^{N} \frac{1}{\sum_{n=1}^{N} w^{(n)}_{0:t}} h(x_{0:t}^{(i)}) w^{(n)}_{0:t}$$
      

Of course, this will require that we be able to evaluate 

$$\pi(x_{0:t}^{(n)}| y_{0:t-1})$$ 

which itself involves some crazy normalization constant, and _the next_ resulting quantity depends on some crazy normalization constant, and so forth. The way to manage this is to start at $$t=0$$ , in which we _can_ evaluate $$\pi(x_0^{(n)})$$ and work recursively to build up to higher dimensions (more on this in future posts).

{% include lib/mathjax.html %}
