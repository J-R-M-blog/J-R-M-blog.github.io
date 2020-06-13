---
layout: default
title: Modeling Non-Linear Time Series With Particle Filtering Part I
---

This post follows the work in the (awesome) paper found at http://www.irisa.fr/aspi/legland/ref/cappe07a.pdf. I condense a couple of their sections to get at the general idea and implement a standard bootstrap particle filter in Python. The goal of this post is to review the derivation of the general state space model (i.e. the posterior distribution and prediction distribution). 

# Problem setup

The following looks like a lot, but it is simply PDF factorizations and Bayes' rule over and over. Also, notice that the below likelihood factorization appears in hidden Markov models (HMMs) as well (where $$X_{t}$$ can take on only a finite number of values).  Anyways, suppose we have two processes $$\{X_{t}\}$$, $$\{Y_{t}\}$$, where $$X_{t}$$ denotes the state of a dynamic system at time $$t$$ and $$Y_{t}$$ denotes the measurement observed at time $$t$$. Suppose the system evolves according to the probability density functions (PDFs):

$$
\begin{aligned}
X_{0} &\sim f(x_{0}) \\
X_{t} &\sim f(x_{t} | x_{t-1}) \\
Y_{t} &\sim g(y_{t} | x_{t})
\end{aligned}
$$

For those familiar, notice that under some further assumptions we arrive at the Kalman filter. 


Okay, so now suppose we have some measurements up to time $$t$$. Define the following 

$$ y_{0:t} \triangleq (y_{0}, y_{1}, \ldots, y_{t}), x_{0:t} \triangleq (x_{0}, x_{1}, \ldots, x_{t}) $$

The joint PDF is given by

$$
\begin{aligned}
\pi(y_{0:t}, x_{0:t})&= \pi(y_{0},y_{1},\ldots,y_{t},x_{0},x_{1},\ldots,x_{t}) \\
\\
&= g(y_{t} | y_{0},\ldots,y_{t-1}, x_{0}, \ldots, x_{t} )f(x_{t} | y_{0},\ldots,y_{t-1}, x_{0}, \ldots, x_{t-1}) \\
\\
&\times g(y_{t-1} | y_{0},\ldots,y_{t-2}, x_{0}, \ldots, x_{t-1} ) f(x_{t-1} | y_{0},\ldots,y_{t-1}, x_{0}, \ldots, x_{t-2}) \\
\\
&\times g(y_{t-2} | y_{0},\ldots,y_{t-3}, x_{0}, \ldots, x_{t-2} ) f(x_{t-2} | y_{0},\ldots,y_{t-1}, x_{0}, \ldots, x_{t-3}) \\
&\vdots \\
&\times g(y_{1} | y_{0}, x_{0}, x_{1}) f(x_{1} | y_{0}, x_{0}) g(y_{0} | x_{0}) f(x_{0})
\\
\\
\end{aligned}
\\
$$

Now apply the fact that $$y_{t}$$ depends only on $$x_{t}$$, that $$x_{t}$$ depends only on $$x_{t-1}$$, etc. This is why we need this assumption! Otherwise the above would (probably) be intractable. 

$$
\\
\begin{aligned}
\pi(y_{0:t}, x_{0:t}) &= g(y_{t} |x_{t} )f(x_{t} | x_{t-1})g(y_{t-1} | x_{t-1} ) f(x_{t-1} | x_{t-2}) \ldots g(y_{1} | x_{1}) f(x_{1} | x_{0}) g(y_{0} | x_{0}) f(x_{0}) \\
&= f(x_{0})g(y_{0}|x_{0}) \prod_{j=1}^{t} f(x_{j} | x_{j-1})g(y_{j}| x_{j})
\\
\end{aligned}
$$

Most people skip to that last line; hopefully, the derivation above explains why.

Typically, the goal is to infer the _true_ state $$X_{t}$$ having only observed $$Y_{t}$$. In this case,

$$
\begin{aligned}
\\
\pi(x_{0:t} | y_{0:t}) &= \frac{\pi(x_{0:t}, y_{0:t})}{\int_{X_{0:t}}\pi(x_{0:t}, y_{0:t}) dx_{0:t} } \\
\\
&= \frac{g(y_{t} | x_{t})f(x_{t}| x_{t-1})f(x_{0})g(y_{0}|x_{0}) \prod_{j=1}^{t-1} f(x_{j} | x_{j-1})g(y_{j}| x_{j})}
{\int_{X_{0:t}}\pi(x_{0:t}, y_{0:t}) dx_{0:t}} \\
\\
&= \frac{g(y_{t} | x_{t})\pi(x_{0:t}, y_{0:t-1})}{\int_{X_{0:t}}\pi(x_{0:t}, y_{0:t}) dx_{0:t}} \\
\\
&= \frac{g(y_{t} | x_{t})\pi(x_{0:t}, y_{0:t-1})}{Z_{t}}
\\
\end{aligned}
$$

where 

$$Z_{t} \triangleq \int_{X_{0:t}}\pi(x_{0:t}, y_{0:t}) dx_{0:t}$$, 

the normalizing constant (constant with respect to $$x_{0:t}$$). 

Notice that 

$$Z_{t} = \pi(y_{0:t}) = \pi(y_{t} | y_{0:t-1})\pi(y_{0:t-1})$$

So we can rewrite the final line above as 

$$
\begin{aligned}
\frac{g(y_{t} | x_{t})\pi(x_{0:t}, y_{0:t})}{Z_{t}} &= \frac{g(y_{t} | x_{t})\pi(x_{0:t}, y_{0:t-1})}{\pi(y_{t} | y_{0:t-1})\pi(y_{0:t-1})} \\
\\
&= \frac{g(y_{t} | x_{t})\pi(x_{0:t}| y_{0:t-1})}{\pi(y_{t} | y_{0:t-1})}
\end{aligned}
$$

The prediction of the future state is given by

$$
\pi(x_{t+1}, x_{0:t} | y_{0:t}) = \pi(x_{0:t}, y_{0:t})f(x_{t+1} | x_{t})
$$

I'll end this post with the following example:

$$
\begin{aligned}
f(x_{t} | x_{t-1}) &= \mathcal{N}\left(x_{t} |  \frac{x_{t-1}}{2} + 25 \frac{x_{t-1}}{1+x^{2}_{t-1}} + 8 \cos(1.2 t), \sigma_{u}\right) \\
g(y_{t} | x_{t}) &= \mathcal{N}\left(y_{t} | \frac{x^{2}_{t}}{20}, \sigma_{v}\right)
\end{aligned}
$$

where $$\sigma_{u},\sigma_{v} \in \mathbb{R}^{+}$$ are assumed known. This system is simulated in Python below. Within the next few posts, we will be able to (hopefully) estimate the state (blue line) having only observed the measurement sequence (orange line). Note that this is an example of a non-linear time series that cannot be modeled using traditional methods.




```python
# Returns the state mean value from the function above
def x_mean(x_prev,t):
    return x_prev/2 + 25*(x_prev)/(1+x_prev**2) + 8*math.cos(1.2*t)

# Returns the measurement mean value from the function above
def y_mean(x_new):
    return x_new**2 / 20
# Draws random normals according to the transition equations
def draw_sample(t,x_prev,system_variance,state_variance):
    x_new = np.random.normal(x_mean(x_prev,t),math.sqrt(state_variance),1)
    y_new = np.random.normal(y_mean(x_new), math.sqrt(system_variance),1)
    return np.array([x_new, y_new]).reshape(2,)

# I set the system, or "measurement", variance low and the state variance large
# This is what the authors do in the linked PDF 
system_var = 1
state_var = 10

# Create a Pandas dataframe
df = pd.DataFrame(columns=('State', 'Measurement'))

# Draws from the prior distribution
df.loc[0] = draw_sample(0,np.random.normal(0,10,1),system_var,state_var)

# Loop through and simulate a draw recursively 
for i in range(1,100):
    df.loc[i] = draw_sample(i,df.at[i-1,'State'],10,1)

# Plots the dataframe created above
df.plot()
```
![Image description](/images/output_1_1.png)

{% include lib/mathjax.html %}
