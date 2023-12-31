## Biases in the distribution of primes in intervals

See [examples.md](https://github.com/tmfreiberg/primes_in_intervals/blob/main/examples.md) for more examples.

<a id='tldr'></a>
### TL;DR
<sup>Jump to: ↓↓ [Contents](#contents) | ↓ [Introduction](#introduction) </sup>

The histogram shows the distribution of primes in intervals. Cramér's random model leads to a prediction for this distribution, shown in orange. We have another prediction, shown in green. In the data we've looked at, as here, our prediction seems to fare better than the one based on Cramér's model. 

As suggested in [[1]](#references), our prediction is based on the Hardy-Littlewood prime tuples conjecture, inclusion-exclusion, and a precise estimate, due to Montgomery and Soundararajan [[3]](#references), for a certain average involving the singular series of the prime tuples conjecture. 

![SegmentLocal](images/README/N_exp21_H_60.gif)

The prediction of Cramér's random model (orange) is, with $\lambda/H = 1/(\log N - 1)$ being the "probability" that an integer close to $N$ is prime (and we assume $\lambda \asymp 1$), is the Binomial distribution $\mathrm{Binom}(H,\lambda/H)$, whose probability mass function is given by

$$f(m; H,\lambda/H) = \binom{H}{m}\left(\frac{\lambda}{H}\right)^m\left(1 - \frac{\lambda}{H}\right)^{H - m} =  \frac{e^{-\lambda}\lambda^m}{m!}\bigg[1 + \frac{Q_1(\lambda,m)}{H} + \frac{Q_2(\lambda,m)}{H^2} + \cdots\bigg],$$
where each $Q_j(\lambda,m)$ is a polynomial in $\lambda$ and $m$, and in particular, 

$$Q_1(\lambda,m) = \frac{m - (m - \lambda)^2}{2}.$$

Our prediction (green) is 

$$F(H,m,\lambda) = \frac{e^{-\lambda}\lambda^m}{m!}\left[1 + \frac{Q_1(\lambda,m)}{H}\left(\log H + (\log 2\pi + \gamma - 1)\right) \right],$$

in agreement with Cramér's model only as a first-order approximation. The secondary term in our prediction is more in line with our observation that the distribution of the numerical data is more "pinched up" around the center: there is more of a _bias_ towards the mean $\lambda$ than is suggested by the Binomial distribution. (Note that $F(H,m,\lambda)$ is not a probability mass function as its sum over $m$ is only close to $1$, but not equal to $1$. $F(H,m,\lambda)$ is an _approximation_ to a probability mass function, viz. the probability that the interval $(a, a + H]$ of length $H$, $a$ chosen uniformly at random from $(N - M, N + M]$, contains exactly $m$ primes. Whether one can find modifications to the higher order terms in the expansion of $f(m; H,\lambda/H)$, that lead to even better predictions for the distribution of primes in intervals, is certainly a tantalizing question, worth pursuing.)

<a id='introduction'></a>
### Introduction
<sup>Jump to: ↓ [Contents](#contents) | ↑ [TL;DR](#tldr)</sup>

Excerpt from letter from Gauss to his student Johann Franz Encke, December 24, 1849 (see [[4]](#references)).

_In 1811, the appearance of Chernau's cribrum gave me much pleasure and I have frequently (since I lack the patience
for a continuous count) spent an idle quarter of an hour to count another chiliad here and there; although I eventually gave it up without quite getting through a million. Only some time later did I make use of the diligence of **Goldschmidt** to fill some of the remaining gaps in the first million and to continue the computation according to Burkhardt’s tables. Thus (for many years now) the first three million have been counted and checked against the integral. A small excerpt follows..._

![SegmentLocal](images/README/nachlass.jpg)

Gauss considered intervals $[0,100), [100,200), [200,300), \ldots$, which he called "centades" (a decade is a 10-year period from '0 to '9, a centade is a 100-year period from '00 to '99). He (and Goldschmidt) kept a tally of the number of centades that contain exactly $m$ primes, for $m = 0, 1, \ldots$. In the table above, we see the data for primes between $2$ and $3$ million. The row labels are $m = 0,1,2,\ldots,17$. Then we have ten columns, the first for the range $200 \times 10^5$ to $210 \times 10^5$, then second for the range $210 \times 10^5$ to $220 \times 10^5$, and so on. In row $m$, column $n$, we have the number of centades between $(200 + 10(n - 1))\times 10^5$ and $(200 + 10n)\times 10^5$ that contain exactly $m$ primes. The additional column on the right shows the row sums, giving the total number of centades between $2$ and $3$ million that contain a given number of primes. The weighted sum over this extra column, where the term corresponding to row $m$ is weighted by $m$, gives the total number of primes between $2$ and $3$ million.

What this allowed Gauss to notice is that the density of primes around $n$ is approximately $1/\log n$. For instance, we see that the average number of primes in a centade between $2$ and $3$ million is between $6$ and $7$, as is $100/\log n$ for $n$ between $2$ and $3$ million. Thus, Gauss conjectured that the number of primes up to $N$ ought to be well approximated by integrating over this density:

$$\pi(N) = \\#\\{p \le N : p \textrm{ prime}\\} \approx \int_2^N \frac{dt}{\log t}.$$

We can see that Gauss/Goldschmidt were comparing their data against this prediction. Gauss's conjecture would become the prime number theorem in 1896, arguably the pinnacle of 19th century mathematics, when Hadamard and de la Vallée Poussin completed the program outlined by Riemann in his seminal 1859 manuscript. In fact, if the Riemann hypothesis is true, then 

$$\pi(N) = \int_2^N \frac{dt}{\log t} + O\left(\sqrt{N}\log N\right).$$

Incidentally, here is the corrected table from Gauss's _Nachlass_.

![SegmentLocal](images/README/goldschmidt_2_3_million.png)

We can see that Gauss/Goldschmidt were only short by $21$ primes in the end: not bad!

What we are interested in here is not just the average number of primes in a centade, but the distribution of primes in centades.

![SegmentLocal](images/README/goldschmidt_table_plot.png)

What proportion of centades between $2$ and $3$ million have exactly $7$ primes, for instance? For such questions we can only conjecture an answer, for they seem to be well beyond current methods at present. Harald Cramér, a Swedish mathematician, statistician, and actuary, put forward a random model for the primes, that has led to many very fascinating and deep conjectures about the distribution of prime numbers. Cramér's model gives us a conjectural answer to our question about the distribution of primes in intervals. 

Let's say we are looking at primes in intervals of the form $(a, a + H]$, for $N - M < a \le N + M$, where $N$ is very large and $M$ is relatively small. By Gauss's observation, confirmed by the prime number theorem, the density of primes around $a$ is well-approximated by $1/\log a$, which, since $a$ is very close to $N$, is well-approximated by $1/\log N$. Cramér interpreted this density as a probability. Thus, what we ask for is the probability that the interval $(a, a + H]$ contains exactly $m$ primes, when the probability that an integer chosen at random from the range in which the interval lies is prime, is close to $1/\log N$. If we think of the indicator function of the primes in $(a, a + H]$ as the outcome of a sequence of $H$ independent Bernoulli trials, with probability of a successful trial (the integer is prime) as $1/\log N$, then we should expect the probability of the interval containing $m$ primes to be approximated by the probability of $m$ successful, and $H - m$ unsuccessful, Bernoulli trials, viz.

$$\binom{H}{m}\left(\frac{1}{\log N}\right)\left(1 - \frac{1}{\log N}\right)^{H - m}.$$

The setup we're interested in here is when $H$ is of order $\log N$ (the average size of the gap between consecutive primes around $N$). Thus, if $H = \lambda \log N$, say, with $\lambda > 0$ constant, the above can be written as

$$\binom{H}{m}\left(\frac{\lambda}{H}\right)\left(1 - \frac{\lambda}{H}\right)^{H - m},$$

which, by the Poisson limit theorem, is asymptotically equal to 

$$\frac{e^{-\lambda}\lambda^m}{m!}$$

as $N \to \infty$.

Now, there is a glaring problem with this heuristic, and that is related to the word _independent_. The event of an integer $n$ being prime is certainly not independent of the event that $n + 1$ is prime, for instance. (If $n > 2$ is prime then it is odd and so $n + 1$, being even, is certainly not prime.) Well, the Hardy-Littlewood prime $k$-tuples conjecture asserts, roughly speaking and continuing in our probabilistic vein, that the probability of a $k$-tuple $n + h_1,\ldots, n + h_k$ of integers all being prime is close to 

$$\frac{\mathfrak{S}(h_1,\ldots,h_k)}{(\log N)^k},$$

if $n$ is very close to $N$. Here $\mathfrak{S}(h_1,\ldots,h_k)$ is a certain function of $h_1,\ldots,h_k$, called the _singular series_ for the $k$-tuple. Too our point about non-independence, $\mathfrak{S}(0,1) = 0$, and $\mathfrak{S}(h_1,\ldots,h_k) = 0$ whenever there is no chance of $n + h_1,\ldots,n + h_k$ all being prime. (As another example, $\mathfrak{S}(0,2,4) = 0$, because one of the integers $n, n + 2, n + 4$ is always a multiple of $3$.) 

The naive application of Cramér's model, which assumes independence, is basically asserting that the singular series is always equal to $1$, which is wrong. However, it turns out that the singular series is very close to $1$ _on average_, in the sense that

$$\sum_{0 < h_1 < \cdots < h_k \le H} \mathfrak{S}(h_1,\ldots,h_k) \sim \sum_{0 < h_1 < \cdots < h_k \le H} 1 \quad (H \to \infty).$$

This turns out to be enough to allow us to conclude, conditionally on a certain form of the Hardy-Littlewood prime $k$-tuples conjecture, that 

$$\frac{1}{N} \\#\\{ a \le N : \pi(a + H) - \pi(a) = m \\} \sim \frac{e^{-\lambda}\lambda^m}{m!} \quad (N \to \infty),$$

where $\lambda > 0$ and $m \ge 0$ are fixed, and where $H = \lambda \log N$. This vindicates Cramér's model: primes are distributed in short intervals essentially as if according to a Poisson process.

The result about the singular series, and the deduction of the conditional Poisson distribution for primes in short intervals, is due to Gallagher [[2]](#references), who used the method of moments. In [[1]](#references) we prove a generalization of this conditional result, using not the method of moments, but an inclusion-exclusion argument, that allows us to directly insert the sum over the singular series into the estimation of the probability mass function. Thus, we are able to make use of more precise estimates for the singular series average, such as the one due to Montgomery and Soundararajan [[3]](#references), leading to more precise predictions with secondary terms (although we need to make further conjectures about the uniformity of the estimate of Montgomery and Soundararajan, or perhaps establish some unconditional results involving further averaging).

The resulting predictions for the distribution of primes in short intervals, agree with that of Cramér's model (and the conditional result of Gallagher) as first-order approximations, but deviate at the second-order term, and appear to better fit the numerical data, where it seems the expected number of primes in an interval is much more popular of a value than is suggested by Cramér's model. The purpose of the code below is to generate data for primes in intervals, compare it to theoretical predictions, and visualize the results in tables and plots.

<a id='contents'></a>
### Contents
<sup>Jump to: ↑ [Introduction](#introduction) | ↓ [Libraries](#libraries) </sup>

[TL;DR](#tldr)<br>
[Contents](#contents)<br>
[Introduction](#introduction)<br>
[Libraries](#libraries)<br>
[Generate](#generate)<br>
..........[Sieve](#sieve)<br>
..........[Count](#count)<br>
..........[Disjoint intervals](#disjoint)<br>
..........[Disjoint intervals, with checkpoints](#disjoint_checkpoints)<br>
..........[Overlapping intervals](#overlapping)<br>
..........[Overlapping intervals, with checkpoints](#overlapping_checkpoints)<br>
..........[Prime-starting intervals, and a more general function](#prime_starting)<br>
..........[A single function](#single_function)<br>
..........[To do](#to_do)<br>
[Raw data](#raw_data)<br> .......... [Example 1](#eg1generate) | [Example 2](#eg2generate)<br>
[Save](#save)<br> .......... [Example 1](#eg1save) | [Example 2](#eg2save)<br>
[Retrieve](#retrieve)<br> .......... [Example 1](#eg1retrieve) | [Example 2](#eg2retrieve)<br>
[Narrow or filter](#narrow)<br> .......... [Example 1](#eg1narrow) | [Example 2](#eg2narrow)<br>
[Partition](#partition)<br> .......... [Example 1](#eg1partition) | [Example 2](#eg2partition)<br>
[Nested intervals](#nest)<br> .......... [Example 1](#eg1nest) | [Example 2](#eg2nest)<br>
[Analyze](#analyze)<br> .......... [Example 1](#eg1analyze) | [Example 2](#eg2analyze)<br>
[Compare](#compare)<br> .......... [Example 1](#eg1compare) | [Example 2](#eg2compare)<br>
[Display](#display)<br> .......... [Example 1](#eg1display) | [Example 2](#eg2display) | [Example 3](#eg3display) (table from Gauss's _Nachlass_)<br>
[Plot & animate](#plot)<br> .......... [Example 1](#eg1plot) | [Example 2](#eg2plot)<br>
[Worked examples](#worked)<br> .......... [Example 4](#eg4worked) | [Example 5](#eg5worked) | [Example 6](#eg6worked) (nested intervals)<br>
[Extensions](#extensions)<br>
[References](#references)<br>

<a id='libraries'></a>
### Libraries
<sup>Jump to: ↑ [Contents](#contents) | ↓ [Generate](#generate) </sup>

```python
# LIBRARIES
from itertools import count # used in the postponed_sieve prime generator
import sqlite3 # to save and retrieve data from a database
import pandas as pd # to display tables of data
from timeit import default_timer as timer # to see how long certain computations take
import numpy as np # for square-roots, exponentials, logs, etc.
from scipy.special import binom as binom # binom(x,y) = Gamma(x + 1)/[Gamma(y + 1)Gamma(x - y + 1)]
from scipy.special import gamma as gamma # Gamma function
from scipy.stats import norm
import sympy # for the Euler-Mascheroni constant, EulerGamma, in the constant 1 - gamma - log(2*pi) from Montgomery-Soundararajan
import matplotlib.pyplot as plt # for plotting distributions
from matplotlib import animation # for animating sequences of plots
from matplotlib import rc # to help with the animation
from IPython.display import HTML # to save animations
from matplotlib.animation import PillowWriter # to save animations as a gif
```

<a id='generate'></a>
### Generate
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Libraries](#libraries) | ↓ [Sieve](#sieve) </sup>

Here we define a series of functions that will help us generate dictionaries containing information about primes in intervals. In the end, we will use the two functions ```disjoint_cp``` and ```overlap_cp``` ([disjoint](#disjoint) and [overlapping](#overlapping) intervals, with "checkpoints"). We build up to them via the non-checkpoint versions ```disjoint``` and ```overlap```. We test our code on a few examples, confirming that it works as intended.

<a id='sieve'></a>
#### Sieve
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Generate](#generate) | ↓ [Count](#count) </sup>

To count primes, we first need to generate them. We could make our own basic sieve of Eratosthenes, but we want something that is a bit more efficient, without getting too fancy. The pros and cons of numerous prime generators are discussed here [How to implement an efficient infinite generator of prime numbers in Python?](https://stackoverflow.com/questions/2211990/how-to-implement-an-efficient-infinite-generator-of-prime-numbers-in-python)

We took the following generator posted by Will Ness on the above Stack Overflow page.

```python
# Prime generator found here:
# https://stackoverflow.com/questions/2211990/how-to-implement-an-efficient-infinite-generator-of-prime-numbers-in-python
# This code was posted by Will Ness. See above URL for further information about who contributed what, 
# and discussion of complexity.
    
from itertools import count
                                         # ideone.com/aVndFM
def postponed_sieve():                   # postponed sieve, by Will Ness      
    yield 2; yield 3; yield 5; yield 7;  # original code David Eppstein, 
    sieve = {}                           #   Alex Martelli, ActiveState Recipe 2002
    ps = postponed_sieve()               # a separate base Primes Supply:
    p = next(ps) and next(ps)            # (3) a Prime to add to dict
    q = p*p                              # (9) its sQuare 
    for r in count(9,2):                 # the Candidate
        if r in sieve:               # r's a multiple of some base prime
            s = sieve.pop(r)         #     i.e. a composite ; or
        elif r < q:  
             yield r                 # a prime
             continue              
        else:   # (r==q):            # or the next base prime's square:
            s=count(q+2*p,2*p)       #    (9+6, by 6 : 15,21,27,33,...)
            p=next(ps)               #    (5)
            q=p*p                    #    (25)
        for m in s:                  # the next multiple 
            if m not in sieve:       # no duplicates
                break
        sieve[m] = s                 # original test entry: ideone.com/WFv4f
```

<a id='count'></a>
#### Count
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Sieve](#sieve) | ↓ [Disjoint intervals](#disjoint) </sup>

We can now count primes. We won't ultimately use the next two functions: we'll just use them to test our 'primes in intervals' counting functions for correctness.

Recall that $$\pi(x) = \\#\\{p \le x : p \text{ prime}\\}.$$ The function ```prime_pi(x,y)``` returns $$\pi(x + y) - \pi(y) = \\#\\{x < p \le y : p \text{ prime}\\}.$$

```python
def next_prime(a): # the first prime after a
    primes = postponed_sieve()
    p = next(primes)
    while p <= a:
        p = next(primes)
    return p

def prime_pi(x,y): # number of primes p such that x < p <= y
    primes = postponed_sieve()
    c = 0
    p  = next(primes)
    while p <= x:
        p = next(primes)
    while p <= y:
        c += 1
        p = next(primes)
    return c
```
```python
# Test the above
next_prime(100), next_prime(101), prime_pi(1,100), prime_pi(1,101)
```
```
(101, 103, 25, 26)
```

<a id='disjoint'></a>
#### Disjoint intervals
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Count](#count) | ↓ [Disjoint intervals, with checkpoints](#disjoint_checkpoints) </sup>

Given $A$, $B$, $H$, and $m$, let 

$$g(m) = \\#\\{1 \le k \le (B - A)/H : \pi(A + kH) - \pi(A + (k - 1)H) = m\\}.$$ 

That is, $g(m)$ of the disjoint intervals 

$$(a, a + H], \quad a = A, A + H, \ldots, A + (K - 1)H$$

contain exactly $m$ primes, where $A + KH \le B < A + (K + 1)H$, i.e. $K = [(B - A)/H]$. 

The function ```disjoint(A,B,H)``` returns a dictionary whose items are of the form ```m : g(m)``` for all ```m``` such that ```g(m)``` is nonzero. 

We just have to iterate over the primes and, as soon as a prime exceeds $A + (k - 1)H$, start a counter, incrementing it by $1$ for each new prime until a prime exceeds $A + kH$. If the counter equals $m$ at this point, we increment the value of our dictionary item ```m : g(m)``` by $1$. We reset the counter and repeat for the next interval, until all intervals have been covered.

```python
# DISJOINT INTERVALS
# Create a dictionary whose items are of the form m : g(m), 
# where g(m) is the number of the disjoint intervals 
# (M, M + H], (M + H, M + 2H],..., (M + (K-1)H, M + KH]
# that contain exactly m primes. Here, M + KH <= N < M + (K + 1)H.

def disjoint(A,B,H):    
    K = (B - A)//H 
    B = A + K*H # re-define A in case the inputs are not of this form    
    output = { m : 0 for m in range(H + 1) } # Initialize the output dictionary covering all possible values for m.
    P = postponed_sieve()
    p = next(P) # initialize p as 2
    a = A # start of the first interval, viz. (A, A + H]
    while p < a + 1:
        p = next(P) # p is now the prime after a (= A initially)
    m = 0 # initialize m as 0        
    for k in range(1, K + 1):
        while p < a + k*H + 1: 
            m += 1
            p = next(P)
        output[m] += 1
        m = 0
    output = { m : output[m] for m in output.keys() if output[m] != 0} # remove m if there are no intervals with m primes       
    return output   
```

```python
# Let's test this a little bit.
# disjoint(0,H, H) should return { m : 1}, where m is the number of primes up to H
for H in range(10,100,10):
    print(H, disjoint(0,H,H), prime_pi(0,H))
```
```
10 {4: 1} 4
20 {8: 1} 8
30 {10: 1} 10
40 {12: 1} 12
50 {15: 1} 15
60 {17: 1} 17
70 {19: 1} 19
80 {22: 1} 22
90 {24: 1} 24
```

```python
test_dict = disjoint(2*10**6,3*10**6,100)
test_dict
```
```
{0: 1,
 1: 25,
 2: 97,
 3: 337,
 4: 776,
 5: 1408,
 6: 1881,
 7: 1995,
 8: 1525,
 9: 1035,
 10: 559,
 11: 227,
 12: 98,
 13: 28,
 14: 6,
 15: 1,
 17: 1}
```

```python
# The values in this dictionary should sum to the number of intervals considered, namely, (3 - 2)*10**6/10**2 = 10**4.
# Secondly, if we sum m*g(m) over all keys m, we should get the total number of primes between 2*10**6 and 3*10**6.
sum([v for v in test_dict.values()]), sum([k*v for k,v in zip(test_dict.keys(),test_dict.values())]), prime_pi(2*10**6,3*10**6)
```
```
(10000, 67883, 67883)
```

<a id='disjoint_checkpoints'></a>
#### Disjoint intervals, with checkpoints
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Disjoint intervals](#disjoint) | ↓ [Overlapping intervals](#overlapping) </sup>

Suppose we are interested in ```disjoint(2*10**6,3*10**6,100)```. For about the same cost, we can compute ```disjoint(2*10**6,2*10**6 + k*10**5,100)``` for ```k = 1,2,...,10```. One motivation for this is for [animating plots](#plots). The function ```disjoint_cp``` takes a list ```C``` and interval length ```H``` as input, and returns a "meta-dictionary" consisting of the items

```C[k] : disjoint(C[0],C[k],H)```

for ```k in range(1,len(C)```. 

Obviously, we don't just compute ```disjoint(C[0],C[k],H)``` for each ```k```, for that would waste a lot of time running through the primes multiple times.

We call the ```C[k]``` "checkpoints".

We'll also add the trivial item ```C[0] : { 0 : 0, 1 : 0, ...}``` to our meta-dictionary. (Note that ```disjoint(C[0],C[0],H)``` returns an empty dictionary ```{}```, not a dictionary with zero-value items.)

In fact, we'll put all of these items into a dictionary, and make this dictionary the value of an item in our ultimate output dictionary, whose key will be ```'data'```. 

At the start of our "meta-dictionary", we will insert the dictionary-valued item 

```'header' : {'interval_type' : 'disjoint', 'lower_bound' : C[0], 'upper_bound' : C[-1], 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : ['data'] } ``` 

to help us identify what it contains. (We'll add more to the meta-dictionary later, expanding the ```'contents'``` as we go along.) We call this item's value the "header" of the data.

Our final output will therefore take the form 

```{ 'header' : { ... }, 'data' : { C[0] : { 0 : 0, 1 : 0, ... }, C[1] : { ... }, ... , C[-1] : { ... } }  }```

In the item ```C[k] : { ... }```, the value is a dictionary consisting of items ```m : g(m)```, where ```g(m)``` is the number of intervals of the form $(a, a + H]$, where $C[0] < a \le C[k]$ and $a = C[0] + jH$ for some integer $j \ge 0$, that contain exactly $m$ primes. 

```python
# ANCILLARY FUNCTION: REMOVE/ADD ZERO-ITEMS FROM/TO META-DICTIONARY

def zeros(meta_dictionary, pad='yes'):
    output = {}
    if pad == 'no':
        for k in meta_dictionary.keys():
            output[k] = { m : meta_dictionary[k][m] for m in meta_dictionary[k].keys() if meta_dictionary[k][m] != 0}
        return output
    padding = set()
    for k in meta_dictionary.keys():        
        padding = padding.union([m for m in meta_dictionary[k] if meta_dictionary[k][m] != 0])
    padding = list(padding)
    padding.sort() 
    for k in meta_dictionary.keys():
        output[k] = {}
        for m in padding:
            if m in meta_dictionary[k].keys():
                output[k][m] = meta_dictionary[k][m]
            else: 
                output[k][m] = 0
    return output
```

```python
# DISJOINT INTERVALS WITH CHECKPOINTS
# Same as above disjoint function, but we want to input a list C = [C_0,...,C_n] of "checkpoints",
# and return disjoint(C_0, C_i, H) for i = 1,...,n.
# We'll put each such dictionary into a "meta-dictionary" whose keys are the C_i.
# We could just compute disjoint(C_0,C_i,H) for each i, but then we'd be re-running
# the prime generator from scratch for each i, and generating primes is by far 
# the most expensive part of this computation (time-wise and memory-wise). 
# We can do what we want by iterating over the primes just once.

def disjoint_cp(C,H):    
    P = postponed_sieve()
    p = next(P)
    # If, e.g., H = 100 and C = [0,10,100,210,350,400], then we replace C by N = [0,100,200,300,400]...
    K, N = [], []
    for i in range(len(C)):
        K.append((C[i] - C[0])//H)
        N.append(C[0] + K[i]*H)
    # Could have repeated elements: in above e.g., K = [0,0,1,2,3,4] and N = [0,0,100,200,300,400], whence
    K = list(set(K))
    N = list(set(N))
    K.sort()
    N.sort()
    output = { 'header' : {'interval_type' : 'disjoint', 'lower_bound' : N[0], 'upper_bound' : N[-1], 'interval_length' : H, 'no_of_checkpoints' : len(N), 'contents' : [] } }
    # OK now N = [0,100,200,300,400] in our e.g., and [N_0, N_0 + K_1*H,...,N_0 + K_n*H] in general.
    data = {}
    for n in N:
        data[n] = {}
    data[N[0]] = { m : 0 for m in range(H + 1) } 
    for i in range(1,len(N)):
        for m in data[N[i - 1]].keys():
            data[N[i]][m] = data[N[i-1]][m]                
        while p < N[i-1] + 1:
            p = next(P)  
        m = 0      
        for k in range(1, (N[i] - N[i - 1])//H + 1):
            while p < N[i - 1] + k*H + 1: 
                m += 1
                p = next(P)
            data[N[i]][m] += 1
            m = 0
    trimmed_data = zeros(data)  
    output['data'] = trimmed_data
    output['header']['contents'].append('data')
    return output   
```

```python
# Let's test this out
C = list(range(2*10**6, 3*10**6 + 10**5,10**5))
H = 100
test_dict_cp = disjoint_cp(C,100)
test_dict_cp
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 11,
  'contents': ['data']},
 'data': {2000000: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0,
   14: 0,
   15: 0,
   17: 0},
  2100000: {0: 0,
   1: 3,
   2: 10,
   3: 32,
   4: 69,
   5: 119,
   6: 198,
   7: 203,
   8: 158,
   9: 114,
   10: 63,
   11: 21,
   12: 8,
   13: 2,
   14: 0,
   15: 0,
   17: 0},...
```
```python
# test_dict_cp['data'][30*10**6] should be the same as test_dict from before.
test_dict_cp['data'][3000000] == test_dict
```
```
True
```

```python
# The following should be the same, except that the checkpoint dictionary will contain items with zero-value.
disjoint(2*10**6,2*10**6 + 10**5,100), test_dict_cp['data'][2*10**6 + 10**5]
```
```
({1: 3,
  2: 10,
  3: 32,
  4: 69,
  5: 119,
  6: 198,
  7: 203,
  8: 158,
  9: 114,
  10: 63,
  11: 21,
  12: 8,
  13: 2},
 {0: 0,
  1: 3,
  2: 10,
  3: 32,
  4: 69,
  5: 119,
  6: 198,
  7: 203,
  8: 158,
  9: 114,
  10: 63,
  11: 21,
  12: 8,
  13: 2,
  14: 0,
  15: 0,
  17: 0})
```

```python
disjoint(2*10**6,2*10**6 + 10**5,100) == zeros(test_dict_cp['data'],pad='no')[2*10**6 + 10**5]
```
```
True
```

```python
# Let's check the agreement for all 10 checkpoints
agree = True
for k in range(1,11):
    agree = agree and (disjoint(2*10**6,2*10**6 + k*10**5,100) == zeros(test_dict_cp['data'],pad='no')[2*10**6 + k*10**5])
agree
```
```
True
```

```python
# Let's test the behaviour for checkpoints that don't really make sense
H = 100
C = [0,50,150,200,650]
test_dict_cp2 = disjoint_cp(C,H)
test_dict_cp2, disjoint(0,600,H)
```
```
({'header': {'interval_type': 'disjoint',
   'lower_bound': 0,
   'upper_bound': 600,
   'interval_length': 100,
   'no_of_checkpoints': 4,
   'contents': ['data']},
  'data': {0: {14: 0, 16: 0, 17: 0, 21: 0, 25: 0},
   100: {14: 0, 16: 0, 17: 0, 21: 0, 25: 1},
   200: {14: 0, 16: 0, 17: 0, 21: 1, 25: 1},
   600: {14: 1, 16: 2, 17: 1, 21: 1, 25: 1}}},
 {14: 1, 16: 2, 17: 1, 21: 1, 25: 1})
```

<a id='overlapping'></a>
#### Overlapping intervals
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Disjoint intervals, with checkpoints](#disjoint_checkpoints) | ↓ [Overlapping intervals, with checkpoints](#overlapping_checkpoints) </sup>

Given $A$, $B$, $H$, and $m$, let 

$$h(m) = \\#\\{A < a \le B : \pi(a + H) - \pi(a) = m\\}.$$ 

That is, $h(m)$ of the $B - A$ overlapping intervals 

$$(A + 1, A + 1 + H], \ldots, (A, A + H]$$ 

contain exactly $m$ primes. The function ```overlap(A,B,H)``` returns a dictionary whose items are of the form ```m : h(m)``` for all ```m``` such that ```h(m)``` is nonzero. 

The idea is to notice that 

$$\pi(a + 1 + H) - \pi(a + 1) = \pi(a + H) - \pi(a) + \mathbb{1}\_{\mathbf{P}}(a + 1 + H) - \mathbb{1}\_{\mathbf{P}}(a + 1),$$

$\mathbb{1}\_{\mathbf{P}}$ being the characteristic function of the primes $\mathbf{P}$. This means that the count $m$ of the number of primes in an interval only changes when an endpoint is prime. We use a "sliding window" of width $H$, and we'll need to generate two sets of primes: one set for the left endpoint, and another for the right. As generating primes is the most time-consuming part of what we're doing here, we expect our ```overlap``` function to take twice as long as our ```disjoint``` function, given the same inputs.

The code is simple, but there is a little bit of fiddling around involved when $a$ gets close to $B$ (specifically, when the prime following $a$ is bigger than $B$).

```python
# OVERLAPPING INTERVALS
# Create a dictionary whose items are of the form m : h(m), 
# where h(m) is the number of a in (M, N] for which the interval
# (a, a + H] contains exactly m primes.
# Note that this means the first interval considered is (M + 1, M + 1 + H],
# and the first prime found is at least M + 2.

def overlap(A,B,H):
    P = postponed_sieve() # We'll need two prime generators (see below).
    Q = postponed_sieve()
    output = { m : 0 for m in range(H + 1) } # Initialize the output dictionary covering all possible values for m.
    a = A + 1 # start of the first interval, viz. (A + 1, A + 1 + H]
    p, q = next(P), next(Q) # initialize p and q as 2
    while p < a + 1:
        p, q = next(P), next(Q) # p and q are now the prime after a (= A + 1 initially)
    m = 0 # initialize m as 0
    while q < a + H + 1: 
        m += 1
        q = next(Q) 
    # q is now the prime after a + H
    # m is the number of primes in our first interval (a, a + H] = (A + 1, A + 1 + H]
    # From now on, imagine a sliding window of length H, starting at a. We have m primes in the window. 
    # Move the window one to the right. If the left endpoint is prime while the right endpoint is not, we lose a prime: m -> m - 1.
    # If the right endpoint is prime while the left is not, we gain a prime: m -> m + 1.
    # Otherwise, m remains unchanged. 
    # Thus, we only need to do update our dictionary when either the left or right endpoint passes a prime.
    # E.g. if the next prime after a is p = a + 10 and the next prime after a + H is q = a + H + 12, 
    # then (a', a' + H] contains m primes for a' = a, a + 1, a + 9, so we can just update our m-counter by nine.
    # Also, (a + 10, a + 10 + H] now contains m - 1 primes. 
    # We'd let p = a + 10 become the new a, m - 1 the new m, p_next the new p, q remains the same, etc.
    # Of course, we have a small problem if a' + 10 exceeds B, so we treat that with a separate loop at the end.
    while p < B + 1:
        output[m] += 1    
        b, c = p - a, q - (a + H) # p = a + b, q = a + H + c
        output[m] = output[m] + min(b,c) - 1
        if b == c:
            a = p
            p = next(P)
        if b < c:
            a, m = p, m - 1
            p = next(P)
        if c < b:
            a, m = a + c, m + 1
        while q < a + H + 1:
            q = next(Q)
    while a < B + 1: # now the prime after a is also bigger than B
        output[m] += 1
        b, c = p - a, q - (a + H) # p = a + b, q = a + H + c
        if a + min(b,c) > B:  
            output[m] = output[m] + B - a
            break
        else: # must be that c < b, because p = a + b > B. 
            output[m] = output[m] + c - 1
            a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)
    output = { m : output[m] for m in output.keys() if output[m] != 0}
    return output
```

```python
# Let's test this out a bit

# a = 1, interval (1, 1 + 5] contains 3 primes
# a = 2, interval (2, 2 + 5] contains 3 primes
# a = 2, interval (3, 3 + 5] contains 2 primes
# a = 3, interval (4, 4 + 5] contains 2 primes 
# a = 4, interval (5, 5 + 5] contains 1 prime

# Hence we should get {1 : 1, 2 : 3, 3: 2} here...
overlap(0,5,5)
```
```
{1: 1, 2: 2, 3: 2}
```

```python
# overlap(M, M + 1, H) should just return the number of primes in (M + 1, M + 1 + H]
M = 999
H = 1000
overlap_test_dict = overlap(M,M + 1,H)
overlap_test_dict, prime_pi(M + 1,M + 1 + H)
```
```
({135: 1}, 135)
```

```python
M = 0
N = 100
H = 10
overlap_test_dict2 = overlap(M,N,H)
overlap_test_dict2, sum([v for v in overlap_test_dict2.values()]), sum([k*v for k,v in zip(overlap_test_dict2.keys(), overlap_test_dict2.values())]) 
# First sum should equal N - M. Last value should be approx but not exactly H*[pi(N) - pi(M)]
```
```
({1: 8, 2: 46, 3: 38, 4: 7, 5: 1}, 100, 247)
```

<a id='overlapping_checkpoints'></a>
#### Overlapping intervals, with checkpoints
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Overlapping intervals](#overlapping) | ↓ [Prime-starting intervals, and a more general function](#prime_starting) </sup>

Analogous to [disjoint intervals, with checkpoints](#disjoint_checkpoints).

```python
# OVERLAPPING INTERVALS, WITH CHECKPOINTS
# Input interval length H and a list of integers C = [N_0,N_1,...,N_n]
# This function will return a dictionary whose keys are N_0,...,N_n.
# The value corresponding to each key will itself be a dictionary.
# The dictionary corresponding to N_k will be { m : h(m), m = 0,...,H},
# where h(m) is the number of intervals (a, a + H], with N_0 < a <= N_k, 
# that contain exactly m primes.

def overlap_cp(C,H):
    output = { 'header' : {'interval_type' : 'overlap', 'lower_bound' : C[0], 'upper_bound' : C[-1], 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : []} }
    data = { C[0] : { m : 0 for m in range(H + 1) } }      
    P = postponed_sieve()  
    Q = postponed_sieve()     
    p, q = next(P), next(Q)  
    m = 0
    current_data = { m : 0 for m in range(H + 1)}
    for i in range(1,len(C)):
        M, N = C[i-1], C[i]
        a = M + 1
        while p < a + 1:
            m -= 1
            p = next(P)   
        while q < a + H + 1: 
            m += 1
            q = next(Q)
        while p < N + 1:                        
            current_data[m] += 1    
            b, c = p - a, q - (a + H)  
            current_data[m] = current_data[m] + min(b,c) - 1
            if b == c:
                a = p
                p = next(P)
            if b < c:
                a, m = p, m - 1
                p = next(P)
            if c < b:
                a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)        
        while a < N + 1: 
            current_data[m] += 1
            b, c = p - a, q - (a + H)  
            if a + min(b,c) > N:  
                current_data[m] = current_data[m] + N - a 
                data[N] = {}
                for k in current_data.keys():
                    data[N][k] = current_data[k]
                break
            else:   
                current_data[m] = current_data[m] + c - 1
                a, m = a + c, m + 1
                while q < a + H + 1:
                    q = next(Q)
    trimmed_data = zeros(data)
    output['data'] = trimmed_data
    output['header']['contents'].append('data')
    return output
```

```python
M = 0
N = 100
H = 10
C = list(range(M, N + 1))
overlap_cp_test_dict = overlap_cp(C,H)
for c in C[47:]:
    print(c, overlap_cp_test_dict['data'][c], overlap_cp_test_dict['data'][c] == overlap(M,c,H)) # should be the same if there are no zero-values in the latter
```
```
47 {1: 1, 2: 18, 3: 22, 4: 5, 5: 1} True
48 {1: 2, 2: 18, 3: 22, 4: 5, 5: 1} True
49 {1: 2, 2: 19, 3: 22, 4: 5, 5: 1} True
50 {1: 2, 2: 20, 3: 22, 4: 5, 5: 1} True
51 {1: 2, 2: 20, 3: 23, 4: 5, 5: 1} True
52 {1: 2, 2: 20, 3: 24, 4: 5, 5: 1} True
...
```

```python
[overlap_cp_test_dict['data'][c][3] for c in C[5:]] # should be increasing
```
```
[1,
 2,
 3,
 4,
 4,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 10,
 ...
```

```python
H = 100
C = list(range(0,100 + 1,10)) # = [0, 10, ..., 100]
another_test_olap_cp = overlap_cp(C, H)
another_test_olap_cp
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 0,
  'upper_bound': 100,
  'interval_length': 100,
  'no_of_checkpoints': 11,
  'contents': ['data']},
 'data': {0: {18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0},
  10: {18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 4, 25: 5, 26: 1},
  20: {18: 0, 19: 0, 20: 0, 21: 0, 22: 2, 23: 2, 24: 10, 25: 5, 26: 1},
  30: {18: 0, 19: 0, 20: 0, 21: 6, 22: 6, 23: 2, 24: 10, 25: 5, 26: 1},
  40: {18: 0, 19: 0, 20: 0, 21: 14, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  50: {18: 0, 19: 2, 20: 6, 21: 16, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  60: {18: 0, 19: 2, 20: 12, 21: 20, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  70: {18: 0, 19: 4, 20: 20, 21: 20, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  80: {18: 0, 19: 14, 20: 20, 21: 20, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  90: {18: 2, 19: 20, 20: 22, 21: 20, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1},
  100: {18: 2, 19: 22, 20: 28, 21: 22, 22: 8, 23: 2, 24: 10, 25: 5, 26: 1}}}
```

```python
import pandas as pd
another_test_olap_cp_df = pd.DataFrame.from_dict(another_test_olap_cp['data'], orient='index').astype('int')#
another_test_olap_cp_df
# The sum of each row should be equal to the label of that row.
# The values in a given column should be increasing as we go down.
```
```
18	19	20	21	22	23	24	25	26
0	0	0	0	0	0	0	0	0	0
10	0	0	0	0	0	0	4	5	1
20	0	0	0	0	2	2	10	5	1
30	0	0	0	6	6	2	10	5	1
40	0	0	0	14	8	2	10	5	1
50	0	2	6	16	8	2	10	5	1
60	0	2	12	20	8	2	10	5	1
70	0	4	20	20	8	2	10	5	1
80	0	14	20	20	8	2	10	5	1
90	2	20	22	20	8	2	10	5	1
100	2	22	28	22	8	2	10	5	1
```

<a id='prime_starting'></a>
#### Prime-starting intervals, and a more general function
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Overlapping intervals, with checkpoints](#prime_starting) | ↓ [A single function](#single_function) </sup>


We have considered intervals of the form $(a, a + H]$, where $a$ runs over integers in an arithmetic progression modulo $H$ (disjoint intervals), and where are runs over all integers (overlapping intervals). We wish to consider intervals for which $a$ is always prime. The basic function is the following.

```python
def prime_start(M,N,H):
    P = postponed_sieve()
    Q = postponed_sieve()
    p = next(P)
    q = next(Q)
    while p <= M:
        p = next(P)
    while q <= p:
        q = next(Q) 
    output = { m : 0 for m in range(H + 1) }
    m = 0
    while p <= N:
        while q <= p + H:
            m += 1
            q = next(Q)
        output[m] += 1
        p = next(P)
        m += -1
    output = { m : output[m] for m in range(H + 1) if output[m] != 0}
    return output
```

```python
prime_start(10,20,20)
# (11,31] has 13, 17, 19, 23, 29, 31
# (13,33] has 17, 19, 23, 29, 31
# (17,37] has 19, 23, 29, 31, 37
# (19,39] has 23, 29, 31, 37
# Therefore should return {4: 1, 5: 2, 6: 1}
```
```
{4: 1, 5: 2, 6: 1}
```

```python
from timeit import default_timer as timer
start = timer()
prime_start_test = prime_start(2000000,3000000,100)
end = timer()
end - start
```
```
1.6957543999888003
```
```python
prime_start_test
```
```
{0: 12,
 1: 155,
 2: 799,
 3: 2584,
 4: 6063,
 5: 10259,
 6: 13359,
 7: 12896,
 8: 10312,
 9: 6468,
 10: 3175,
 11: 1283,
 12: 396,
 13: 97,
 14: 15,
 15: 5,
 16: 2,
 17: 3}
```

Here is the version with checkpoints.

```python
def prime_start_cp(C,H):
    C.sort()
    P = postponed_sieve()
    Q = postponed_sieve()
    p = next(P)
    q = next(Q)    
    output = { 'header' : {'interval_type' : 'prime_start', 'lower_bound' : C[0], 'upper_bound' : C[-1], 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : []} }
    data = { C[0] : { m: 0 for m in range(H + 1)} }
    current = { m : 0 for m in range(H + 1) }
    m = 0
    while p <= C[0]:
        p = next(P)
    while q <= p:
        q = next(Q) 
    for i in range(len(C)):
        M, N = C[i - 1], C[i]          
        while p <= N:
            while q <= p + H:
                m += 1
                q = next(Q)
            current[m] += 1            
            p = next(P)
            m += -1
        data[N] = {}
        for k in range(H + 1):
            data[N][k] = current[k]
    trimmed_data = zeros(data)
    output['data'] = trimmed_data
    output['header']['contents'].append('data')
    return output
```
```python
start = timer()
prime_start_cp_test = prime_start_cp(list(range(2000000,3000001,100000)),100)
end = timer()
end - start
```
```
1.715596199966967
```
```python
prime_start_cp_test['data'][3000000] == prime_start_test
```
```
True
```

In fact, why not consider $a$ running over any strictly increasing sequence $A$ of nonnegative integers? For that matter, why restrict ourselves to primes in intervals? Why don't we consider the number of intervals with $m$ elements from another strictly increasing sequence $B$ of nonnegative integers? If we can generate $A$ and $B$, we can count intervals $(a, a + H]$ with $a$ running over $A$, containing a given number of elements of $B$. This can be done with ```anyIntervals``` below.

```python
def anyIntervals(M,N,H,generator1,generator2):
    A = generator1
    B = generator2
    a = next(A)
    b = next(B)
    while a <= M:
        a = next(A)    
    output = { m : 0 for m in range(H + 1) }
    m = 0
    Blist = []
    while a <= N:   
	while b <= a:
            b = next(B)	    
        while b <= a + H:
            m += 1
            Blist.append(b)
            b = next(B)
        output[m] += 1
        a = next(A)
        temp_m = m
        for i in range(temp_m):
            if Blist[0] <= a: 
                m += -1
                Blist.pop(0)
    output = { m : output[m] for m in range(H + 1) if output[m] != 0}
    return output
```

For the case of overlapping intervals and primes, we'd use ```count()``` from the ```itertools``` package as ```generator1```, and ```postponed_sieve()``` as ```generator2```. Let's try it out.

```python
from timeit import default_timer as timer
start1 = timer()
nachlass_general_way = anyIntervals(2*10**6, 3*10**6,100, count(), postponed_sieve())
end1 = timer()

start2 = timer()
nachlass_specific_way = pii.overlap_cp([2*10**6, 3*10**6],100)
end2 = timer()

end1 - start1, end2 - start2
```
```
(2.5513191999998526, 1.858088900000439)
```
The greater generality costs us a little bit extra in time. (We also have to store a list whose length is at most $H$.) Let's check our results for consistency.

```python
nachlass_general_way == nachlass_specific_way['data'][3000000]
```
```
True

```

What about disjoint intervals? That's just considering $(a, a + H]$ where $a$ is in an arithmetic progression $\bmod H$. ```count(b,n)``` from ```itertools``` gives us the arithmetic progression $b \bmod n$.

```python
from timeit import default_timer as timer
start1 = timer()
nachlass_general_way = anyIntervals(2*10**6-100, 3*10**6-100,100, count(0,100), postponed_sieve())
end1 = timer()

start2 = timer()
nachlass_specific_way = pii.disjoint_cp([2*10**6, 3*10**6],100)
end2 = timer()

end1 - start1, end2 - start2
```
```
(0.902533600004972, 0.9400128999986919)
```
```python
nachlass_general_way == nachlass_specific_way['data'][3000000]
```
```
True
```

Notice that the input was ```anyIntervals(2*10**6-100, 3*10**6-100,100, count(0,100), postponed_sieve())```, i.e. we subtracted $H = 100$ from the lower and upper bounds. This is because ```anyIntervals(M,N,H, count(0,H), postponed_sieve())``` gives us data on intervals $(a, a + 100]$ for $M < a \le N$ with $a \equiv 0 \bmod H$. If we'd put $M = 2\cdot 10^6$ and $N = 3\cdot 10^6$ into the function, it would have returned data for $(a, a + 100]$ for $a = 2\cdot 10^6 + 100,2\cdot 10^6 + 200,\ldots,3\cdot 10^6$, where as the ```disjoint_cp``` function gives us data for $a = 2\cdot 10^6,2\cdot 10^6 + 100,\ldots,3\cdot 10^6 - 100$.

Let's see the ```anyIntervals``` function applied to intervals whose left endpoints are all prime. For purposes of demonstration, let's add two lines to the code so that we see a print out at each key step.

```python
def anyIntervalsPrint(M,N,H,generator1,generator2):
    A = generator1
    B = generator2
    a = next(A)
    b = next(B)
    while a <= M:
        a = next(A)
    output = { m : 0 for m in range(H + 1) }
    m = 0
    Blist = []
    while a <= N:  
	while b <= a:
            b = next(B)	    
        while b <= a + H:
            m += 1
            Blist.append(b)
            b = next(B)
        output[m] += 1
	print(f'{a} < {Blist} <= {a + H}, {m} : {output[m]}')
        a = next(A)
        temp_m = m
        for i in range(temp_m):
            if Blist[0] <= a: 
                m += -1
                Blist.pop(0)
    output = { m : output[m] for m in range(H + 1) if output[m] != 0}
    return output
```

```python
anyIntervalsPrint(10,31,10,postponed_sieve(),postponed_sieve())
```
```
11 < [13, 17, 19] <= 21, 3 : 1
13 < [17, 19, 23] <= 23, 3 : 2
17 < [19, 23] <= 27, 2 : 1
19 < [23, 29] <= 29, 2 : 2
23 < [29, 31] <= 33, 2 : 3
29 < [31, 37] <= 39, 2 : 4
31 < [37, 41] <= 41, 2 : 5
{2: 5, 3: 2}
```

<a id='single_function'></a>
#### A single function
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Prime-starting intervals, and a more general function](#prime_starting) | ↓ [To do](#to_do) </sup>

```python
def intervals(C,H,interval_type='overlap'):
    # interval_type is either 'disjoint' or 'prime-start' or not (defaults to 'overlap' unless 'disjoint'/'prime-start' is explicitly given).
    if interval_type == 'disjoint':
        return disjoint_cp(C,H)
    # if interval_type is 'prime_start' or not given or is anything string other than 'disjoint'
    if interval_type == 'prime_start': 
        return prime_start_cp(C,H)
    # if interval_type is 'overlap' or not given or is anything string other than 'disjoint' or 'prime_start'
    if interval_type == 'overlap': 
        return overlap_cp(C,H)
```

<a id='to_do'></a>
#### To do
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [A single function](#single_function) | ↓ [Raw data](#raw_data) </sup>

* A checkpoint version of our ```anyIntervals``` function, plus have the output include a ```header``` etc.

* If we do a computation that takes a long time, and then want to extend the calculation, we'd like to be able to pick up where we left off. Suppose we compute ```intervals([0,N],H)``` where ```N``` is very large, and then we'd like to compute ```intervals([N,2N], H)``` or ```intervals([0,2N], H)```. At the moment, we'd have to start from scratch. What we'd like to do is save the state of our ```intervals``` function, particularly the prime generators, and then just keep going.

* In a similar vein, another thing we could do is input various values for the interval length ```H```, say a list ```[H_1,H_2,...,H_k]```, and have a function return ```intervals(C,H_i)``` for each ```H_i```, without simply computing ```intervals(C,H_i)``` $k$ times.

* Some of the code for displaying data (below) is getting unwieldy, and should be cleaned up. 

<a id='raw_data'></a>
### Raw data
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [To do](#to_do) | ↓ [Save](#save) </sup>

Now let's generate some actual data to work with.

<a id='eg1generate'></a>
#### Example 1.

```python
A1 = 2*10**6 # lower bound
B1 = 3*10**6 # upper bound
H1 = 100 # interval length
step1 = 10**4 # increments for checkpoints
C1 = list(range(A1,B1 + 1,step1)) # checkpoints
```

```python
# Let's see how long some computations take as well
from timeit import default_timer as timer
start = timer()
disjoint(C1[0], C1[-1],H1)
end = timer()
end - start
```
```
0.8698457999853417
```

```python
start = timer()
data1disj = intervals(C1,H1,'disjoint') # same as disjoint_cp(C1,H1)
end = timer()
end - start
# 0.8928492000559345, not much longer than without "checkpoints".
```
```
0.8928492000559345
```

```python
start = timer()
overlap(C1[0],C1[-1],H1)
end = timer()
end - start
# 1.9600014999741688 about double the time required for the disjoint version
```
```
1.9600014999741688
```

```python
start = timer()
data1olap = intervals(C1,H1,'overlap') # same as overlap_cp(C1,H1)
end = timer()
end - start
# 1.9077042000135407, about the same as the non-checkpoint version, and about double the disjoint analog
```
```
1.9077042000135407
```

<a id='eg2generate'></a>
#### Example 2.

```python
# Let's look at numbers that are a bit larger
N2 = int(np.exp(18))
H2 = 85
step2 = 12*H2 # = 1020 (increments for checkpoints)
M2 = 100*step2 # = 1200*H2 = 102,000
A2 = N2 - M2 # lower bound
B2 = N2 + M2 # upper bound
C2 = list(range(A2, B2 + 1, step2)) # checkpoints. Note that these do not have to be at regular intervals.
```
```python
start = timer()
data2disj = intervals(C2,H2,'disjoint') # same as disjoint_cp(C2,H2)
end = timer()
end - start
```
```
20.59070379997138
```

```python
start = timer()
data2olap = intervals(C2,H2,'overlap') # same as overlap_cp(C2,H2)
end = timer()
end - start
# 40.94606200000271, twice as long as 'disjoint' analog, as expected
```
```
40.94606200000271
```

<a id='save'></a>
### Save
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Raw data](#raw_data) | ↓ [Retrieve](#retrieve) </sup>

We won't want to re-do a computation that takes a lot of time and memory, so we store our data in a database for future use.

```python
# We'll create two database tables for primes in intervals: one for disjoint counts and one for overlapping counts.
# First column shall be A (lower bound), second column B (upper bound), third column H (interval length), 
# where we consider intervals of the form (a, a + H] for a in (A,B].
# These three columns will constitute the table's primary key.
# The next max_primes columns shall contain the number of such intervals with m primes, 
# where m = 0,1,...,max_primes will easily cover any situation we will be interested in, where

import sqlite3

max_primes = 100 # can change this and alter tables in future if need be

# To save some typing...
cols = ''
for i in range(max_primes + 1):
    cols = cols + 'm' + f'{i}' + ' int, '

conn = sqlite3.connect('primes_in_intervals_db')
conn.execute('CREATE TABLE IF NOT EXISTS disjoint_raw (lower_bound int, upper_bound int, interval_length int,' + cols + 'PRIMARY KEY(lower_bound, upper_bound, interval_length))')
conn.execute('CREATE TABLE IF NOT EXISTS overlap_raw (lower_bound int, upper_bound int, interval_length int,' + cols + 'PRIMARY KEY(lower_bound, upper_bound, interval_length))')
conn.execute('CREATE TABLE IF NOT EXISTS prime_start_raw (lower_bound int, upper_bound int, interval_length int,' + cols + 'PRIMARY KEY(lower_bound, upper_bound, interval_length))')
conn.commit()
conn.close()

# So, A, B, H, m0, m1, ..., m[max_primes] are columns 0, 1, 2, 3, ..., max_primes + 3, respectively: mi is column i + 3.
```

```python
# In case we mess up and need to start again, but 
# BE CAREFUL if the table contains data from a calculation that took a long time.
# conn = sqlite3.connect('primes_in_intervals_db')
# conn.execute('DROP TABLE IF EXISTS disjoint_raw')
# conn.execute('DROP TABLE IF EXISTS overlap_raw')
# conn.execute('DROP TABLE IF EXISTS prime_start_raw')
# conn.close()
```

```python
# Store the data in our database table.

import sqlite3

def save(data): 
    if 'data' not in data.keys():
        return print('No data to save. Check contents.')
    C = list(data['data'].keys())
    H = data['header']['interval_length']
    # We'll insert rows of the form C[0], C[k], H, g(0), g(1), ..., g(max_primes)
    # into our disjoint_raw table, and the same with h in place of g in our overlap_raw table.
    # Thus, there are max_primes + 3 columns total. For the SQL string...
    qstring = ''
    for i in range(max_primes + 4):
        qstring += '?,'
    qstring = qstring[:-1]
    conn = sqlite3.connect('primes_in_intervals_db')
    for k in range(1,len(C)):
        row = [0]*(max_primes + 4)
        row[0], row[1], row[2] = C[0], C[k], H
        for m in data['data'][C[k]].keys():
            row[m + 3] = data['data'][C[k]][m]
        if data['header']['interval_type'] == 'disjoint':
            conn.executemany('INSERT OR IGNORE INTO disjoint_raw VALUES(' + qstring + ')', [tuple(row)])
        if data['header']['interval_type'] == 'overlap':
            conn.executemany('INSERT OR IGNORE INTO overlap_raw VALUES(' + qstring + ')', [tuple(row)])
	if data['header']['interval_type'] == 'prime_start':
            conn.executemany('INSERT OR IGNORE INTO prime_start_raw VALUES(' + qstring + ')', [tuple(row)]) 		
    conn.commit()
    conn.close()
```

<a id='eg1save'></a>
#### Example 1.

```python
save(data1disj)
save(data1olap)
```

<a id='eg2save'></a>
#### Example 1.

```python
save(data2disj)
save(data2olap)
```

<a id='retrieve'></a>
### Retrieve
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Save](#save) | ↓ [Narrow](#narrow) </sup>

```python
# First, we'll define a function that shows an entire table in our database.
# After this, we'll define a function that takes H as an input and 
# reconstructs the original dictionary(ies) we created that correspond to interval length H.

import sqlite3
import pandas as pd

def show_table(interval_type, description='description'):
    conn = sqlite3.connect('primes_in_intervals_db')
    c = conn.cursor()    
    if interval_type == 'disjoint':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='disjoint_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for disjoint intervals.')
            return
        else:
            res = conn.execute("SELECT * FROM disjoint_raw ORDER BY lower_bound ASC, upper_bound ASC, interval_length ASC")
    if interval_type == 'overlap':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='overlap_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for overlapping intervals.')
            return
        else:        
            res = conn.execute("SELECT * FROM overlap_raw ORDER BY lower_bound ASC, upper_bound ASC, interval_length ASC")  
    if interval_type == 'prime_start':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prime_start_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for prime-starting intervals.')
            return
        else:        
            res = conn.execute("SELECT * FROM prime_start_raw ORDER BY lower_bound ASC, upper_bound ASC, interval_length ASC")  		
    rows = res.fetchall()
    c.close()
    conn.close()
    cols = ['A', 'B', 'H']
    for m in range(0,max_primes + 1):
        cols.append(m)
    df = pd.DataFrame(rows, columns = cols)        
    if description == 'no description':
           return df
    else:
        if interval_type == 'disjoint':
            return df.style.set_caption('Disjoint intervals. ' + r'Column with label $m$ shows $\#\{1 \le k \le (B - A)/H : \pi(A + kH) - \pi(A + (k - 1)H) = m \}$')
        if interval_type == 'overlap':
            return df.style.set_caption('Overlapping intervals. ' + r'Column with label $m$ shows $\#\{A < a \le B : \pi(a + H) - \pi(a) = m \}$')
	if interval_type == 'prime_start':
            return df.style.set_caption('Prime-starting intervals. ' + r'Column with label $m$ shows $\#\{A < p \le B : \pi(p + H) - \pi(p) = m \}$, $p$ prime.')	
```

```python
show_table('disjoint')
```
```
Disjoint intervals. Column with label  𝑚
  shows  #{1≤𝑘≤(𝐵−𝐴)/𝐻:𝜋(𝐴+𝑘𝐻)−𝜋(𝐴+(𝑘−1)𝐻)=𝑚}
 
 	A	B	H	0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63	64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84	85	86	87	88	89	90	91	92	93	94	95	96	97	98	99	100
0	2000000	2010000	100	0	0	0	3	7	13	25	10	13	16	10	2	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
1	2000000	2020000	100	0	0	0	6	15	23	45	32	31	30	12	3	2	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
2	2000000	2030000	100	0	0	2	11	24	32	58	55	46	41	20	8	2	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
3	2000000	2040000	100	0	0	4	13	28	47	74	80	68	45	26	11	2	2	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
...
```

```python
show_table('overlap', description='no description').head() # it's getting large and takes a while to display
```
```
A	B	H	0	1	2	3	4	5	6	...	91	92	93	94	95	96	97	98	99	100
0	2000000	2010000	100	0	0	38	314	586	1250	1962	...	0	0	0	0	0	0	0	0	0	0
1	2000000	2020000	100	48	12	58	450	1164	2718	3860	...	0	0	0	0	0	0	0	0	0	0
2	2000000	2030000	100	48	12	180	888	1968	3808	5466	...	0	0	0	0	0	0	0	0	0	0
3	2000000	2040000	100	48	40	280	1048	2500	5092	7588	...	0	0	0	0	0	0	0	0	0	0
4	2000000	2050000	100	48	78	388	1290	3404	6440	9540	...	0	0	0	0	0	0	0	0	0	0
5 rows × 104 columns
```

```python
# Retrieve data from our database table. Recall that we have the row
# C[0], C[k], H, g(0), g(1), ..., g(max_primes)
# in our disjoint_raw table (similarly in our overlap_raw table). 
# We want to reconstruct the dictionary 
# {'signature' : {'interval_type' : 'disjoint/overlap', 'count' : 'cumulative', 'lower_bound' : A, 'upper_bound' : B, 'interval_length' : H},  
#'data' : { C[0] : {m : g(m), ...}, C[1] : { m : g(m), ... } }  }

import sqlite3

def retrieve(H, interval_type = 'overlap'):
    conn = sqlite3.connect('primes_in_intervals_db')
    c = conn.cursor()    
    if interval_type == 'disjoint':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='disjoint_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for disjoint intervals.')
            return
        else:
            res = conn.execute("SELECT * FROM disjoint_raw WHERE (interval_length) = (?) ORDER BY lower_bound ASC, upper_bound ASC", (H,))
    if interval_type == 'overlap':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='overlap_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for overlapping intervals.')
            return
        else:        
            res = conn.execute("SELECT * FROM overlap_raw WHERE (interval_length) = (?) ORDER BY lower_bound ASC, upper_bound ASC", (H,))
     if interval_type == 'prime_start':
        existence_check = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prime_start_raw'").fetchall()
        if existence_check == []:
            print('Database contains no table for prime-starting intervals.')
            return
        else:        
            res = conn.execute("SELECT * FROM prime_start_raw WHERE (interval_length) = (?) ORDER BY lower_bound ASC, upper_bound ASC", (H,))      
    rows = res.fetchall()
    #rows = [(C[0], C[k], H, g(0), ..., g(100)), k = 0,1,...), (C'[0], C'[k], H, g(0),...,g(100)), k = 0,1,...),...]
    c.close()
    conn.close()
    found = {}
    i = 0
    while i < len(rows):        
        A = rows[i][0] # C[0]
        found[A] = {} 
        j = i
        while j < len(rows) and rows[j][0] == A:
            B = rows[j][1]
            found[A][B] =  { m - 3 : rows[j][m] for m in range(3,max_primes + 4) } 
            j += 1
        i = j 
    output = []
    for A in found.keys():
        C = list(found[A].keys())
        C.insert(0,A)
        outputA = { 'header' : {'interval_type' : interval_type, 'lower_bound' : A, 'upper_bound' : C[-1], 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : ['data'] } }        
        data = { C[0] : { m : 0 for m in range(H + 1)} }
        for c in C[1:]:
            data[c] = found[A][c]
        trimmed_data = zeros(data)
        outputA['data'] = trimmed_data
        output.append(outputA)
    if len(output) == 1:
        print(f'Found {len(output)} dataset corresponding to interval of length {H} ({interval_type} intervals).')
        print(f"\n \'header\' : {output[0]['header']}\n")
        return output[0]        
    else:
        print(f'Found {len(output)} datasets corresponding to interval of length {H} ({interval_type} intervals).')
        for i in range(len(output)):
            print(f"\n [{i}] \'header\' : {output[i]['header']}\n")   
        return output
```

<a id='eg1retrieve'></a>
#### Example 1.

```python
start = timer()
retrieve_data1disj = retrieve(H1,'disjoint')
retrieve_data1olap = retrieve(H1,'overlap')[0]
end = timer()
end - start
# 0.04226740007288754
```
```
Found 1 dataset corresponding to interval of length 100 (disjoint intervals).

 'header' : {'interval_type': 'disjoint', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}

Found 2 datasets corresponding to interval of length 100 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 100, 'no_of_checkpoints': 201, 'contents': ['data']}

0.04226740007288754
```

```python
# Check that the retrieved data is the same as the original data
data1disj == retrieve_data1disj, data1olap == retrieve_data1olap
```
```
(True, True)
```

<a id='eg2retrieve'></a>
#### Example 2.

```python
start = timer()
retrieve_data2disj = retrieve(H2,'disjoint')
retrieve_data2olap = retrieve(H2,'overlap')
end = timer()
end - start
# 0.5290923999855295
```
```
Found 1 dataset corresponding to interval of length 85 (disjoint intervals).

 'header' : {'interval_type': 'disjoint', 'lower_bound': 65557969, 'upper_bound': 65761969, 'interval_length': 85, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 85 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 65557969, 'upper_bound': 65761969, 'interval_length': 85, 'no_of_checkpoints': 201, 'contents': ['data']}

0.5290923999855295
```

```python
# Check that the retrieved data is the same as the original data
data2disj == retrieve_data2disj, data2olap == retrieve_data2olap
```
```
(True, True)
```

<a id='narrow'></a>
### Narrow
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Retrieve](#retrieve) | ↓ [Partition](#partition) </sup>

If we have data on primes in intervals of the form $(a, a + H]$ for $A < a \le B$, and we wish to instead work with data for the more restricted range $C < a \le D$ where $A \le C < D \le B$, we can obtain the corresponding data if $C$ and $D$ are "checkpoints". We use the ```extract``` function below. It's simply a matter of noting that, e.g., if $100$ intervals with $a$ in the range $(A, D]$ contain $5$ primes, while $60$ of them correspond to $a$ in $(A,C]$, then $100 - 60 = 40$ of them are with $a$ in $(C, D]$.

More generally, we might wish to restrict (or "filter") our sequence ```C``` of checkpoints to a subsequence ```D```. We might especially want to do this if we have retrieved some data from our database, as in the following example. Suppose ```H = 100```, ```C1 = [100,200,300,400,500]```, and ```C2 = [100,160,220,280,340]```. We apply ```intervals(C1,H)```, then save the resulting data to our database, the do the same for ```intervals(C2,H)```. We later apply ```retrieve(100)```. We will end up with the equivalent of ```intervals(C,H)```, where ```C = [100,160,200,220,280,300,340,400,500]```. If we look at the terms of ```C``` we would probably realize that our original intention was to have checkpoints at multiples of ```100``` in one dataset, and checkpoints at ```100``` plus multiples of ```60``` in another. We could then split ```C``` up into ```C1``` and ```C2```. We'd then apply the ```extract``` function below.

```python
# Input a dataset and a range (A,B].
# Output a NEW dataset with info about primes in intervals (a, a + H] with a in (A,B] (option = 'narrow'), 
# or with checkpoints common to newC and the current checkpoints (option = 'filter').

def extract(meta_dictionary, newC, option='filter'): 
    # newC is a list. 
    # option is either 'narrow' or not (defaults to 'filter').
    # if option=='narrow', newC should be of the form [A,B] where (A, B] is the desired range for checkpoints.
    # if A and B are already checkpoints, then both 'narrow' and 'filter' will do the same thing.
    if 'data' not in meta_dictionary.keys():
        return print('No data to filter.')
    if option=='narrow':
        if len(newC) != 2:
            return print('To narrow checkpoints to range (A, B], enter list [A,B].')
        oldC = list(meta_dictionary['data'].keys())
        oldC.sort() # just in case: it's important that these are in increasing order
        C = [c for c in oldC if newC[0] <= c <= newC[-1]]
        if len(C) < 2:
            return print('At least two of the new checkpoints must lie in the given range.')
    else:
        oldC = set(meta_dictionary['data'].keys())
        C = list(oldC.intersection(set(newC)))
        C.sort()
        if len(C) < 2:
            return print('At least two of the new checkpoints must coincide with the old checkpoints.')
    interval_type = meta_dictionary['header']['interval_type'] 
    A, B = C[0], C[-1], 
    H = meta_dictionary['header']['interval_length']
    output = {'header' : {'interval_type' : interval_type, 'lower_bound'  : A, 'upper_bound' : B, 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : []} }
    output['data'] = {}
    for c in C:
        output['data'][c] = {}
        for m in meta_dictionary['data'][c].keys():
            output['data'][c][m] = meta_dictionary['data'][c][m] - meta_dictionary['data'][A][m]
    trimmed_data = zeros(output['data'])
    output['data'] = trimmed_data
    output['header']['contents'].append('data')
    return output    
```

<a id='eg1narrow'></a>
#### Example 1.

```python
narrow_data1disj = extract(retrieve_data1disj,[2000000,4000000],option='narrow') # should be the same
narrow_data1disj == retrieve_data1disj
```
```
True
```

```python
narrow_data1olap = extract(retrieve_data1olap,[2345612,2987654], option='narrow')
narrow_data1olap['header'], narrow_data1olap['data'][2980000], intervals([2350000,2980000],100,'overlap')['data'][2980000], narrow_data1olap['data'][2980000] == intervals([2350000,2980000],100,'overlap')['data'][2980000]
```
```
({'interval_type': 'overlap',
  'lower_bound': 2350000,
  'upper_bound': 2980000,
  'interval_length': 100,
  'no_of_checkpoints': 64,
  'contents': ['data']},
 {0: 52,
  1: 1108,
  2: 6274,
  3: 21990,
  4: 51848,
  5: 90386,
  6: 119192,
  7: 118814,
  8: 97960,
  9: 65798,
  10: 34682,
  11: 14982,
  12: 5148,
  13: 1408,
  14: 250,
  15: 62,
  16: 18,
  17: 16,
  18: 12},
 {0: 52,
  1: 1108,
  2: 6274,
  3: 21990,
  4: 51848,
  5: 90386,
  6: 119192,
  7: 118814,
  8: 97960,
  9: 65798,
  10: 34682,
  11: 14982,
  12: 5148,
  13: 1408,
  14: 250,
  15: 62,
  16: 18,
  17: 16,
  18: 12},
 True)
```

```python
oldC = list(retrieve_data1disj['data'].keys())
newC = [oldC[0], oldC[20], oldC[-1]]
filter_data1disj = extract(retrieve_data1disj,newC) #option='filter' by default
filter_data1disj
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 3,
  'contents': ['data']},
 'data': {2000000: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0,
   14: 0,
   15: 0,
   17: 0},
  2200000: {0: 0,
   1: 5,
   2: 19,
   3: 59,
   4: 139,
   5: 264,
   6: 381,
   7: 404,
   8: 325,
   9: 224,
   10: 115,
   11: 39,
   12: 17,
   13: 6,
   14: 3,
   15: 0,
   17: 0},
  3000000: {0: 1,
   1: 25,
   2: 97,
   3: 337,
   4: 776,
   5: 1408,
   6: 1881,
   7: 1995,
   8: 1525,
   9: 1035,
   10: 559,
   11: 227,
   12: 98,
   13: 28,
   14: 6,
   15: 1,
   17: 1}}}
```

<a id='eg2narrow'></a>
#### Example 2.

```python
narrow_data2disj = extract(retrieve_data2disj,[int(np.exp(18)) - 2*10**3,int(np.exp(18)) + 2*10**3],option='narrow')
narrow_data2disj
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 65658949,
  'upper_bound': 65660989,
  'interval_length': 85,
  'no_of_checkpoints': 3,
  'contents': ['data']},
 'data': {65658949: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
  65659969: {2: 0, 3: 5, 4: 1, 5: 2, 6: 3, 7: 1, 8: 0},
  65660989: {2: 1, 3: 7, 4: 2, 5: 5, 6: 5, 7: 3, 8: 1}}}
```

```python
narrow_data2olap = extract(retrieve_data2olap,[int(np.exp(18)) - 2*10**3,int(np.exp(18)) + 2*10**3],option='narrow')
narrow_data2olap
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 65658949,
  'upper_bound': 65660989,
  'interval_length': 85,
  'no_of_checkpoints': 3,
  'contents': ['data']},
 'data': {65658949: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0},
  65659969: {0: 1,
   1: 34,
   2: 81,
   3: 156,
   4: 228,
   5: 293,
   6: 133,
   7: 81,
   8: 13,
   9: 0,
   10: 0},
  65660989: {0: 1,
   1: 34,
   2: 194,
   3: 315,
   4: 390,
   5: 521,
   6: 265,
   7: 163,
   8: 81,
   9: 64,
   10: 12}}}
```

<a id='partition'></a>
### Partition
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Narrow](#narrow) | ↓ [Nested intervals](#nest) </sup>

We may wish to present our data in a "partitioned" form, i.e. if we have data on intervals $(A,C_1], (A, C_2],\ldots, (A,C_k]$, we may wish to express this as data for $(A, C_1], (C_1,C_2],\ldots,(C_{k-1},C_k]$. We do this with the ```partition``` function. We can reverse the process with the ```unpartition``` function.

```python
# Input a dataset and a range (A,B] with checkpoints A = C_0, C_1,..., C_k = B.
# MODIFY the dataset with info about primes in intervals (a, a + H] with a in (C_{i-1},C_i] for i = 1,...,k.

def partition(meta_dictionary):
    if 'data' not in meta_dictionary.keys():
        return print('No data to partition.')
    if 'partition' in meta_dictionary.keys():
        return print('Partitioned data already exists.')
    C = list(meta_dictionary['data'].keys())
    C.sort() # just in case: it's important that these are in increasing order
    partitioned_data = { C[0] : meta_dictionary['data'][C[0]] }
    for k in range(1,len(C)):
        partitioned_data[C[k]] = {}
        for m in meta_dictionary['data'][C[k]].keys():
            partitioned_data[C[k]][m] = meta_dictionary['data'][C[k]][m] - meta_dictionary['data'][C[k - 1]][m] 
    meta_dictionary['partition'] = {}
    for c in C:
        meta_dictionary['partition'][c] = {}
        for m in partitioned_data[c].keys():
            meta_dictionary['partition'][c][m] = partitioned_data[c][m]
    meta_dictionary['header']['contents'].append('partition')
    return meta_dictionary

def unpartition(meta_dictionary):
    if 'partition' not in meta_dictionary.keys():
        return print('No data to unpartition.')
    if 'data' in meta_dictionary.keys():
        return print('Unpartitioned data already exists.')
    C = list(meta_dictionary['partition'].keys())
    C.sort() # just in case: it's important that these are in increasing order
    unpartitioned_data = { C[0] : meta_dictionary['partition'][C[0]] }    
    for k in range(1,len(C)):
        unpartitioned_data[C[k]] = {}
        for m in meta_dictionary['partition'][C[k]].keys():
            unpartitioned_data[C[k]][m] = meta_dictionary['partition'][C[k]][m] + unpartitioned_data[C[k - 1]][m] 
    meta_dictionary['data'] = {}
    for c in C:
        meta_dictionary['data'][c] = {}
        for m in unpartitioned_data[c].keys():
            meta_dictionary['data'][c][m] = unpartitioned_data[c][m]
    meta_dictionary['header']['contents'].append('data')
    return meta_dictionary
```

<a id='eg1partition'></a>
#### Example 1.

```python
retrieve_data1disj = retrieve(100,'disjoint')
```
```
Found 1 dataset corresponding to interval of length 100 (disjoint intervals).

 'header' : {'interval_type': 'disjoint', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}
```

```python
partition(retrieve_data1disj)
retrieve_data1disj['header'], retrieve_data1disj['partition']
```
```
({'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data', 'partition']},
 {2000000: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0,
   14: 0,
   15: 0,
   17: 0},
  2010000: {0: 0,
   1: 0,
   2: 0,
   3: 3,
   4: 7,
   5: 13,
   6: 25,
   7: 10,
   8: 13,
   9: 16,
   10: 10,
   11: 2,
   12: 0,
   13: 1,
   14: 0,
   15: 0,
   17: 0},
   ...
```

```python
partition(retrieve_data1olap)
retrieve_data1olap['header'], retrieve_data1olap['partition']
```
```
({'interval_type': 'overlap',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data', 'partition']},
 {2000000: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0,
   14: 0,
   15: 0,
   16: 0,
   17: 0,
   18: 0},
  2010000: {0: 0,
  ...
```

```python
testunpartition = { 'header' : retrieve_data2olap['header'] }
testunpartition['header']['contents'] = ['partition']
testunpartition['partition'] = retrieve_data2olap['partition']
testunpartition
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['partition']},
 'partition': {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0},
  65558989: {0: 0,
  ...
```

```python
unpartition(testunpartition)
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['partition', 'data']},
 'partition': {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0},
  65558989: {0: 0,
   1: 40,
   ...
```
```python
testunpartition['data'] == retrieve_data2olap['data']
```
```
True
```

<a id='eg2partition'></a>
#### Example 2.

```python
partition(retrieve_data2disj)
retrieve_data2disj['header'], retrieve_data2disj['partition']
```
```
({'interval_type': 'disjoint',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['data', 'partition']},
 {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0},
  65558989: {0: 0,
   1: 1,
   ...
```
   
```python
partition(retrieve_data2olap)
retrieve_data2olap['header'], retrieve_data2olap['partition']
```
```
({'interval_type': 'overlap',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['data', 'partition']},
 {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0},
  65558989: {0: 0,
   1: 40,
   ...
```

<a id='nest'></a>
### Nested intervals
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Partition](#partition) | ↓ [Analyze](#analyze) </sup>

```python
# Input a dataset with data corresponding to checkpoints [C_0,...,C_K], where C_0 < C_1 < ... < C_K. 
# Output a NEW dataset with data corresponding to the intervals (assuming K = 2k + 1 is odd)
# (C_k, C_{k + 1}], (C_{k-1}, C_{k + 2}], ..., (C_0, C_K].
# Note that each interval is contained in the next, so these are "nested" intervals.
# If the C's form an arithmetic progression, then each of the nested intervals share a common midpoint, N (say).
# Thus, the density of primes in these intervals is approximately 1/(log N - 1).
# However, this gets worse and worse as an approximation for all primes in an interval as the interval get wider.
# Thus, there is a trade-off between having a better approximation to the density of primes in an interval, and 
# the number of datapoints (length of the interval).
# It will be interesting to see how these play off against each other.

def nest(dataset):
    if 'data' not in dataset.keys():
        if 'partition' not in dataset.keys():            
            return print('No data to work with, or data is not in a suitable configuration for nesting.')
        else:
            unpartition(dataset)
    C = list(dataset['data'].keys())
    C.sort()
    if len(C) < 3:
        return print('At least three checkpoints needed for a nontrivial nesting.')
    interval_type = dataset['header']['interval_type']
    A = dataset['header']['lower_bound']
    B = dataset['header']['upper_bound']
    H = dataset['header']['interval_length']
    no_of_checkpoints = dataset['header']['no_of_checkpoints']
    nest = { 'header' : {'nested_intervals' : 0, 'interval_type' : interval_type, 'lower_bound': A, 'upper_bound' : B, 'interval_length' : H, 'no_of_checkpoints' : no_of_checkpoints, 'contents' : [] } }
    nest['nested_interval_data'] = {}
    if len(C)%2 == 1:
        C.pop(len(C)//2)
    k = len(C)//2
    M = list(dataset['data'][C[-1]].keys())
    for i in range(k):
        nest['nested_interval_data'][C[k - i - 1], C[k + i]] = {}
        for m in M:
            nest['nested_interval_data'][C[k - i - 1], C[k + i]][m] = dataset['data'][C[k + i]][m] - dataset['data'][C[k - i - 1]][m]
    nest['header']['nested_intervals'] = k
    nest['header']['contents'].append('nested_interval_data')
    return nest
```

<a id='eg1nest'></a>
#### Example 1

```python
get_data1disj = retrieve(100,'disjoint')
```
```
Found 1 dataset corresponding to interval of length 100 (disjoint intervals).

 'header' : {'interval_type': 'disjoint', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}```

```python
nest_data1disj = nest(get_data1disj)
nest_data1disj
```
```
{'header': {'nested_intervals': 50,
  'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['nested_interval_data']},
 'nested_interval_data': {(2490000, 2510000): {0: 0,
   1: 0,
   2: 4,
   3: 6,
   4: 11,
   5: 27,
   6: 44,
   7: 42,
   8: 33,
   9: 21,
   10: 5,
   11: 2,
   12: 3,
   13: 2,
   14: 0,
   15: 0,
   17: 0},
  (2480000, 2520000): {0: 0,
   1: 2,
   2: 8,
   3: 9,
   4: 35,
   ...
```

```python
get_data1olap = retrieve(100,'overlap')[0]
```
```
Found 3 datasets corresponding to interval of length 100 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 100, 'no_of_checkpoints': 201, 'contents': ['data']}


 [2] 'header' : {'interval_type': 'overlap', 'lower_bound': 1318715734, 'upper_bound': 1318915734, 'interval_length': 100, 'no_of_checkpoints': 200, 'contents': ['data']}
```

```python
nest_data1olap = nest(get_data1olap)
nest_data1olap
```
```
{'header': {'nested_intervals': 50,
  'interval_type': 'overlap',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['nested_interval_data']},
 'nested_interval_data': {(2490000, 2510000): {0: 0,
   1: 2,
   2: 180,
   3: 740,
   4: 1350,
   5: 2728,
   6: 4148,
   ...
```

<a id='eg2nest'></a>
#### Example 2

```python
get_data2disj = retrieve(90,'disjoint')
```
```
Found 1 dataset corresponding to interval of length 90 (disjoint intervals).

 'header' : {'interval_type': 'disjoint', 'lower_bound': 65559979, 'upper_bound': 65759959, 'interval_length': 90, 'no_of_checkpoints': 203, 'contents': ['data']}
```

```python
nest_data2disj = nest(get_data2disj)
nest_data2disj
```
```
{'header': {'nested_intervals': 101,
  'interval_type': 'disjoint',
  'lower_bound': 65559979,
  'upper_bound': 65759959,
  'interval_length': 90,
  'no_of_checkpoints': 203,
  'contents': ['nested_interval_data']},
 'nested_interval_data': {(65658979, 65660959): {0: 0,
   1: 1,
   2: 1,
   3: 2,
   4: 4,
   5: 4,
   ...
```

```python
get_data2olap = retrieve(90,'overlap')[0]
```
```
Found 3 datasets corresponding to interval of length 90 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 65559979, 'upper_bound': 65759959, 'interval_length': 90, 'no_of_checkpoints': 203, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 90, 'no_of_checkpoints': 201, 'contents': ['data']}


 [2] 'header' : {'interval_type': 'overlap', 'lower_bound': 1318715734, 'upper_bound': 1318915734, 'interval_length': 90, 'no_of_checkpoints': 200, 'contents': ['data']}
```

```python
nest_data2olap = nest(get_data2olap)
nest_data2olap
```
```
{'header': {'nested_intervals': 101,
  'interval_type': 'overlap',
  'lower_bound': 65559979,
  'upper_bound': 65759959,
  'interval_length': 90,
  'no_of_checkpoints': 203,
  'contents': ['nested_interval_data']},
 'nested_interval_data': {(65658979, 65660959): {0: 0,
   1: 26,
   2: 142,
   3: 271,
   4: 328,
   5: 507,
   6: 322,
   7: 176,
   8: 122,
   9: 44,
   10: 42,
   11: 0,
   12: 0,
   13: 0},
  (65657989, 65661949): {0: 10,
   1: 146,
   2: 386,
   3: 672,
   ...
```

<a id='analyze'></a>
### Analyze
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Nested intervals](#nest) | ↓ [Compare](#compare) </sup>

```python
# ANCILLARY FUNCTION

# Input a dictionary and output a new dictionary, sorted by keys (if the keys are sortable).
def dictionary_sort(dictionary):  
    L = list(dictionary.keys()) 
    L.sort() 
    sorted_dictionary = {}  
    for k in L: 
        sorted_dictionary[k] = dictionary[k] 
    return sorted_dictionary 

# Now some code that will take a dictionary as input.
# Assume the keys are numbers and each key's value is the number of times (frequency of) the key occurs in some data.
# The output is a dictionary, whose first item is itself a dictionary, whose keys are the same as the input dictionary, 
# and for which each key's value is the _proportion_ of occurrences (_relative_ frequency) of the key among the data.
# The other items in the output dictionary are mean, variance, median, mode, etc., of the original data.

import numpy as np

def dictionary_statistics(dictionary): 
    frequencies = dictionary_sort(dictionary)
    relative_frequencies = {} 
    number_of_objects_counted = 0 
    mean = 0 
    median = 0 
    mode = [] 
    second_moment = 0 
    variance = 0 
    standard_deviation = 0 
    M = max(frequencies.values()) 
    for s in frequencies.keys(): 
        number_of_objects_counted += frequencies[s] 
        mean += s*frequencies[s]  
        second_moment += (s**2)*frequencies[s] 
        if frequencies[s] == M:
            mode.append(s) 
    mean = mean/number_of_objects_counted
    second_moment = second_moment/number_of_objects_counted
    variance = second_moment - mean**2 
    standard_deviation = np.sqrt(variance)
    
# A little subroutine for computing the median... 

    temp_counter = 0 
    if number_of_objects_counted%2 == 1: 
        for s in frequencies.keys():
            if temp_counter < number_of_objects_counted/2:
                temp_counter += frequencies[s]
                if temp_counter > number_of_objects_counted/2:
                    median = s
    if number_of_objects_counted%2 == 0: 
        for s in frequencies.keys():
            if temp_counter < number_of_objects_counted/2:
                temp_counter += frequencies[s]
                if temp_counter >= number_of_objects_counted/2:
                    median = s 
        temp_counter = 0 
        for s in frequencies.keys():
            if temp_counter < 1 + (number_of_objects_counted/2):
                temp_counter += frequencies[s]
                if temp_counter >= 1 + (number_of_objects_counted/2):
                    median = (median + s)/2     

# Finally, let's get the relative frequencies.

    for s in frequencies.keys(): 
        relative_frequencies[s] = frequencies[s]/number_of_objects_counted

    output_dictionary = {} 
    output_dictionary["dist"] = relative_frequencies
    output_dictionary["mean"] = mean
    output_dictionary["2ndmom"] = second_moment
    output_dictionary["var"] = variance
    output_dictionary["sdv"] = standard_deviation
    output_dictionary["med"] = median
    output_dictionary["mode"] = mode

    return output_dictionary

# Input a dataset containing a 'data' item, and MODIFY the dataset by 
# adding a new 'distribution' item and a new 'statistics' item, using our 
# ancillary dictionary_statistics function.

def analyze(dataset):
    if 'distribution' in dataset.keys() and 'statistics' in dataset.keys():
        return print('Data has already been analyzed.')
    if 'data' in dataset.keys():    
        C = list(dataset['data'].keys())
        dataset['distribution'] = { C[0] : {} } # no meaningful statistics for the trivial item
        dataset['statistics'] = { C[0] : {} }
        for c in C[1:]:
            temp_dict = dictionary_statistics(dataset['data'][c])
            dataset['distribution'][c] = temp_dict['dist']
            dataset['statistics'][c] = {}
            dataset['statistics'][c]['mean'] = temp_dict['mean']
            dataset['statistics'][c]['2ndmom'] = temp_dict['2ndmom']
            dataset['statistics'][c]['var'] = temp_dict['var']
            dataset['statistics'][c]['sdv'] = temp_dict['sdv']
            dataset['statistics'][c]['med'] = temp_dict['med']
            dataset['statistics'][c]['mode'] = temp_dict['mode']
        dataset['header']['contents'].append('distribution')
        dataset['header']['contents'].append('statistics')
        return dataset
    if 'nested_interval_data' in dataset.keys():    
        C = list(dataset['nested_interval_data'].keys())
        dataset['distribution'] = {  } 
        dataset['statistics'] = {  }
        for c in C:
            temp_dict = dictionary_statistics(dataset['nested_interval_data'][c])
            dataset['distribution'][c] = temp_dict['dist']
            dataset['statistics'][c] = {}
            dataset['statistics'][c]['mean'] = temp_dict['mean']
            dataset['statistics'][c]['2ndmom'] = temp_dict['2ndmom']
            dataset['statistics'][c]['var'] = temp_dict['var']
            dataset['statistics'][c]['sdv'] = temp_dict['sdv']
            dataset['statistics'][c]['med'] = temp_dict['med']
            dataset['statistics'][c]['mode'] = temp_dict['mode']
        dataset['header']['contents'].append('distribution')
        dataset['header']['contents'].append('statistics')
        return dataset
    return print('No data to analyze.')
```

<a id='eg1analyze'></a>
#### Example 1.

```python
analyze(retrieve_data1disj)
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data', 'partition', 'distribution', 'statistics']},
 'data': {2000000: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0,
   12: 0,
   13: 0,
   14: 0,
   15: 0,
   17: 0},
  2010000: {0: 0,
   1: 0,
   2: 0,
   3: 3,
   4: 7,
   5: 13,
   ...
```

```python
retrieve_data1disj['distribution'][2500000]
```
```
{0: 0.0,
 1: 0.0024,
 2: 0.0094,
 3: 0.0316,
 4: 0.0752,
 5: 0.1366,
 6: 0.186,
 7: 0.1986,
 8: 0.1602,
 9: 0.1104,
 10: 0.0538,
 11: 0.0246,
 12: 0.0082,
 13: 0.0024,
 14: 0.0006,
 15: 0.0,
 17: 0.0}
```

```python
retrieve_data1disj['statistics'][2500000]
```
```
{'mean': 6.8278,
 '2ndmom': 50.6258,
 'var': 4.006947160000003,
 'sdv': 2.00173603654428,
 'med': 7.0,
 'mode': [7]}
```

```python
analyze(retrieve_data1olap)
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data', 'partition', 'distribution', 'statistics']},
 'data': {2000000: {0: 0,
   1: 0,
   2: 0,
   ...
```

```python
retrieve_data1disj['distribution'][2500000], retrieve_data1disj['statistics'][2500000]
```
```
({0: 0.0,
  1: 0.0024,
  2: 0.0094,
  3: 0.0316,
  4: 0.0752,
  5: 0.1366,
  6: 0.186,
  7: 0.1986,
  8: 0.1602,
  9: 0.1104,
  10: 0.0538,
  11: 0.0246,
  12: 0.0082,
  13: 0.0024,
  14: 0.0006,
  15: 0.0,
  17: 0.0},
 {'mean': 6.8278,
  '2ndmom': 50.6258,
  'var': 4.006947160000003,
  'sdv': 2.00173603654428,
  'med': 7.0,
  'mode': [7]})
```

```python
analyze(nest_data1disj)
nest_data1disj['distribution']
```
```
{(2490000, 2510000): {0: 0.0,
  1: 0.0,
  2: 0.02,
  3: 0.03,
  4: 0.055,
  5: 0.135,
  6: 0.22,
  7: 0.21,
  8: 0.165,
  9: 0.105,
  10: 0.025,
  11: 0.01,
  12: 0.015,
  13: 0.01,
  14: 0.0,
  15: 0.0,
  17: 0.0},
 (2480000, 2520000): {0: 0.0,
  1: 0.005,
  2: 0.02,
  3: 0.0225,
  4: 0.0875,
  ...
```

```python
analyze(nest_data1olap)
nest_data1olap['statistics'][(2490000, 2510000)]
```
```
{'mean': 6.7601,
 '2ndmom': 49.3741,
 'var': 3.675147989999992,
 'sdv': 1.917067549670588,
 'med': 7.0,
 'mode': [6]}
```

<a id='eg2analyze'></a>
#### Example 2.

```python
analyze(retrieve_data2disj)
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['data', 'partition', 'distribution', 'statistics']},
 'data': {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   4: 0,
   5: 0,
   6: 0,
   7: 0,
   8: 0,
   9: 0,
   10: 0,
   11: 0},
  65558989: {0: 0,
   1: 1,
   2: 1,
   3: 2,
   ...
```

```python
analyze(retrieve_data2olap)
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 65557969,
  'upper_bound': 65761969,
  'interval_length': 85,
  'no_of_checkpoints': 201,
  'contents': ['data', 'partition', 'distribution', 'statistics']},
 'data': {65557969: {0: 0,
   1: 0,
   2: 0,
   3: 0,
   ...
```

```python
analyze(nest_data2disj)
Cnest = list(nest_data2disj['nested_interval_data'].keys())
nest_data2disj['statistics'][Cnest[0]]
```
```
{'mean': 5.045454545454546,
 '2ndmom': 28.40909090909091,
 'var': 2.9524793388429735,
 'sdv': 1.718278015585072,
 'med': 5.0,
 'mode': [7]}
```

```python
analyze(nest_data2olap)
Cnest = list(nest_data2olap['nested_interval_data'].keys())
C[20], nest_data2olap['distribution'][Cnest[20]]
```
```
((1318794734, 1318836734),
 {0: 0.0012987012987012987,
  1: 0.016378066378066377,
  2: 0.06955266955266955,
  3: 0.1303030303030303,
  4: 0.197017797017797,
  5: 0.2342953342953343,
  6: 0.16998556998557,
  7: 0.10134680134680135,
  8: 0.047546897546897546,
  9: 0.023136123136123135,
  10: 0.0075998075998075995,
  11: 0.001443001443001443,
  12: 9.62000962000962e-05,
  13: 0.0})
```

<a id='compare'></a>
### Compare
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Analyze](#analyze) | ↓ [Display](#display) </sup>

Let's work out how many intervals in a range should contain a given number of primes, according to Cramér's model,
and according to our prediction. In place of the actual number, we'll place a quadruple with the actual number, Cramér's prediction, our preferred prediction ($F$), and our alternate prediction ($F^\*$).

Below, ```frei``` is the function 

$$F(H,m,\lambda) = \frac{e^{-\lambda}\lambda^m}{m!}\left[1 + \frac{\log H + (\log 2\pi + \gamma - 1)}{H}\cdot \frac{m - (m - \lambda)^2}{2} \right].$$

and ```frei_alt``` is the function

$$F^\*(H,m,\lambda^\*) = \frac{e^{-\lambda^\*}(\lambda^\*)^m}{m!}\left[1 + \frac{\lambda^\*}{H}(m - \lambda^\*) + \frac{\log H + (\log 2\pi + \gamma - 1)}{H}\cdot \frac{m - (m - \lambda^\*)^2}{2} \right],$$

We use $F$ with $\lambda = H/(\log N - 1)$ and $F^\*$ with $\lambda^\* = H/\log N$, $\lambda/H$ (or $\lambda^\*/H$) representing the "probability" of an integer very close to $N$ being prime.

We're comparing these against the prediction based on $$\mathrm{Binom}(H,\lambda/H)$$.

NB: $F$ and $F^\*$ apply to the case of overlapping intervals only at this point. The details of the case of disjoint intervals and prime-starting intervals have not been worked out yet, and it may well be that different second-order terms arise in the disjoint/prime-starting case. Therefore, in the case of disjoint/prime-starting intervals, comparisons with estimates arising from $F$ and $F^\*$ should be taken with a grain of salt.

```python
import numpy as np 
from scipy.special import binom as binom  
from scipy.special import gamma as gamma  
import sympy 

def binom_pmf(H,m, p):
    return binom(H,m)*(p**m)*(1 - p)**(H - m)

def pois_pmf(H,m,L):
    return (np.exp(-L))*(L**m)/gamma(m + 1)

MS = 1 - sympy.EulerGamma.evalf() - np.log(2*(np.pi)) # "Montgomery-Soundararajan" constant
def frei(H,m,t):
    Q_2 = ((m - t)**2 - m)/2
    return np.exp(-t)*(t**m/gamma(m + 1))*(1 - ((np.log(H) - MS)/(H))*Q_2)

def frei_alt(H,m,t):
    Q_1 = m - t
    Q_2 = ((m - t)**2 - m)/2
    return np.exp(-t)*(t**m/gamma(m + 1))*(1 + (t/H)*Q_1 - ((np.log(H) - MS)/(H))*Q_2)
```

```python
def compare(dataset):
    if 'data' in dataset.keys():
        if 'distribution' not in dataset.keys():
            return print('Analyze data first, to obtain distribution data for comparison with theoretical predictions.')
        C = list(dataset['data'].keys())
        C.sort() # just in case --- this is important
        interval_type = dataset['header']['interval_type']
        A = C[0]
        H = dataset['header']['interval_length']
        comparison = { C[0] : { m : 0 for m in dataset['data'][C[0]].keys() } } # for consistency with the keys
        for c in C[1:]:
            comparison[c] = {}
            N = (A + c)//2 # midpoint of the interval (A, c]
            p = 1/(np.log(N) - 1) # more accurate estimate for the density of primes around (A, c]
            p_alt = 1/np.log(N) # estimate for the density        
            if interval_type == 'overlap':            
                multiplier = c - A # the number of intervals considered, in the overlapping case
            if interval_type == 'disjoint':
                multiplier = (c - A)//H # the number of intervals considered, in the disjoint case
            if interval_type == 'prime_start':
                multiplier = sum(dataset['data'][c].values()) # the number of intervals considered, in the prime-start case
            for m in dataset['data'][c].keys():
                binom_prob = binom_pmf(H,m,p)
                frei_prob = frei(H,m,H*p)
                frei_alt_prob = frei_alt(H,m,H*p_alt)
                binom_pred = int(binom_prob*multiplier) # what dataset['data'][c][m] should be according to Cramer's model
                frei_pred = int(frei_prob*multiplier) # what dataset['data'][c][m] should be up to second-order approximation, at least around the centre of the distribution, according to me, but only in the case of overlapping intervals
                frei_alt_pred = int(frei_alt_prob*multiplier) # the alternative estimate (overlapping intervals)
                comparison[c][m] = (dataset['distribution'][c][m], binom_prob, frei_prob, frei_alt_prob), (dataset['data'][c][m], binom_pred, frei_pred, frei_alt_pred)
        dataset['comparison'] = {}
        for c in C:
            dataset['comparison'][c] = {}
            for m in comparison[c].keys():
                dataset['comparison'][c][m] = comparison[c][m]
        dataset['header']['contents'].append('comparison - actual, binomial, frei, frei_alt')
        return dataset
    if 'nested_interval_data' in dataset.keys():
        if 'distribution' not in dataset.keys():
            return print('Analyze data first, to obtain distribution data for comparison with theoretical predictions.')
        C = list(dataset['nested_interval_data'].keys())
        interval_type = dataset['header']['interval_type']
        H = dataset['header']['interval_length']
        comparison = { } 
        for c in C:
            comparison[c] = {}
            N = (c[0] + c[1])//2 # midpoint of the interval c = (c[0], c[1]].
            p = 1/(np.log(N) - 1) # more accurate estimate for the density of primes around (A, c]
            p_alt = 1/np.log(N) # estimate for the density        
            if interval_type == 'overlap':            
                multiplier = c[1] - c[0] # the number of intervals considered, in the overlapping case
            if interval_type == 'disjoint':
                multiplier = (c[1] - c[0])//H # the number of intervals considered, in the disjoint case
            if interval_type == 'prime_start':
                multiplier = sum(dataset['nested_interval_data'][c].values()) # the number of intervals considered, in the prime-start case
            for m in dataset['nested_interval_data'][c].keys():
                binom_prob = binom_pmf(H,m,p)
                frei_prob = frei(H,m,H*p)
                frei_alt_prob = frei_alt(H,m,H*p_alt)
                binom_pred = int(binom_prob*multiplier) # what dataset['data'][c][m] should be according to Cramer's model
                frei_pred = int(frei_prob*multiplier) # what dataset['data'][c][m] should be up to second-order approximation, at least around the centre of the distribution, according to me, but only in the case of overlapping intervals
                frei_alt_pred = int(frei_alt_prob*multiplier) # alternative prediction (overlapping intervals)
                comparison[c][m] = (dataset['distribution'][c][m], binom_prob, frei_prob, frei_alt_prob), (dataset['nested_interval_data'][c][m], binom_pred, frei_pred, frei_alt_pred)
        dataset['comparison'] = {}
        for c in C:
            dataset['comparison'][c] = {}
            for m in comparison[c].keys():
                dataset['comparison'][c][m] = comparison[c][m]
        dataset['header']['contents'].append('comparison - actual, binomial, frei, frei_alt')        
        return dataset
    if 'data' not in dataset.keys() and 'new_interval_data' not in dataset.keys():
        return print('No data to compare.')
```

We might also like to know which of the predictions fit the data best. But what do we mean by "best"? We mean the prediction $\mathrm{pred}$ for which 

$$\sum_m (h(m) - \mathrm{pred}(m))^2$$

is the smallest. We'll just sum over the smallest range for $m$ outside of which $h(m)$ is zero. (Recall that $h(m)$ is the number of overlapping intervals considered that contain exactly $m$ primes. For disjoitn intervals, replace $h(m)$ by $g(m)$.)

Note that comparing the Binomial with parameters $H$ and $\lambda/H$ is not an apples to apples comparison with $F^\*(H,m,\lambda^\*)$, but we're mainly interested in comparing $F$ and $F^\*$ (we expect both to be superior to the Binomial, regardless of whether we plug $\lambda/H$ or $\lambda^\*/H$ into the Binomial).

```python
# Input a data set containing a 'comparisons' item.
# MODIFY the dataset to add a 'winners' item, giving the "best" prediction for each interval considered in the data.
# "Best" in two different senses: sum of the squared error (over m), and number of m for which a prediction is closest.

def winners(dataset):
    if 'winners' in dataset.keys():
        return print('This function has already been applied to the data.')
    if 'comparison' not in dataset.keys():
        return print('Compare the data first, to obtain distribution data for comparison with theoretical predictions.')
    if 'nested_interval_data' in dataset.keys(): 
        datakey = 'nested_interval_data'
    elif 'data' in dataset.keys():
        datakey = 'data'
    else:
        return print('No data.')
    C = list(dataset[datakey].keys())
    interval_type = dataset['header']['interval_type']
    A = C[0]
    H = dataset['header']['interval_length']
    winners = {}
    for c in C:
        winners[c] = {}
        M = [m for m in dataset['comparison'][c].keys() if dataset['comparison'][c][m] != 0]
        if M != []:
            min_m, max_m = min(M), max(M)
            M = list(range(min_m, max_m + 1))
            square_error_binom = sum([(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][1])**2 for m in M])
            square_error_frei = sum([(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][2])**2 for m in M])
            square_error_frei_alt = sum([(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][3])**2 for m in M])
            winners[c]['B sq error'] = square_error_binom
            winners[c]['F sq error'] = square_error_frei
            winners[c]['F* sq error'] = square_error_frei_alt
            square_error = [square_error_binom, square_error_frei, square_error_frei_alt]
            square_error.sort()
            for i in [0,1,2]:
                if square_error[i] == square_error_frei:
                    winners[c][i + 1] = 'F'
                if square_error[i] == square_error_frei_alt:
                    winners[c][i + 1] = 'F*'
                if square_error[i] == square_error_binom:
                    winners[c][i + 1] = 'B'
            winners[c]['B wins for m in '] = []
            winners[c]['F wins for m in '] = []
            winners[c]['F* wins for m in '] = []
            mB, mF, mFalt = 0, 0, 0
            for m in M:
                temp_list = [abs(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][i]) for i in range(1,4)]
                min_diff = min(temp_list)                
                if abs(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][1]) == min_diff:
                    winners[c]['B wins for m in '].append(m)
                    mB += 1
                if abs(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][2]) == min_diff:
                    winners[c]['F wins for m in '].append(m)
                    mF += 1
                if abs(dataset['comparison'][c][m][1][0] - dataset['comparison'][c][m][1][3]) == min_diff:
                    winners[c]['F* wins for m in '].append(m) 
                    mFalt += 1
            max_wins = [mB, mF, mFalt]
            max_wins.sort(reverse=True)
            winners[c]['most wins'] = ''
            winners[c]['2nd most wins'] = ''
            winners[c]['least wins'] = ''
            if mB == max_wins[0]:
                winners[c]['most wins'] += 'B'
            if mF == max_wins[0]:
                winners[c]['most wins'] += 'F'
            if mFalt == max_wins[0]:
                winners[c]['most wins'] += 'F*'
            if mB == max_wins[1]:
                winners[c]['2nd most wins'] += 'B'
            if mF == max_wins[1]:
                winners[c]['2nd most wins'] += 'F'
            if mFalt == max_wins[1]:
                winners[c]['2nd most wins'] += 'F*'
            if mB == max_wins[2]:
                winners[c]['least wins'] += 'B'
            if mF == max_wins[2]:
                winners[c]['least wins'] += 'F'
            if mFalt == max_wins[2]:
                winners[c]['least wins'] += 'F*'
                
        if M == []:
            winners[c] = {'B sq error' : '-', 'F sq error' : '-', 'F* sq error' : '-', 1 : '-', 2 : '-', 3 : '-', 'B wins for m in ' : '-', 'F wins for m in ' : '-','F* wins for m in ' : '-', 'most wins' : '-', '2nd most wins' : '-', 'least wins' : '-'}
    dataset['winners'] = winners
    dataset['header']['contents'].append('winners')
    return dataset
```

<a id='eg1compare'></a>
#### Example 1.

```python
compare(retrieve_data1disj)
```
```
{'header': {'interval_type': 'disjoint',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data',
   'partition',
   'distribution',
   'statistics',
   'comparison - actual, binomial, frei, frei_alt']},
 'data': {2000000: {0: 0,
 ...
```

```python
# NB: in the disjoint intervals case, our estimates frei and frei_alt are not intended to be applied.
retrieve_data1disj['comparison'][2500000]
```
```
{0: ((0.0,
   0.0004896111528373992,
   -0.000403686551731212,
   -0.000938660440152872),
  (0, 2, -2, -4)),
 1: ((0.0024,
   0.003877665619266112,
   -0.000855326604440596,
   -0.00289519949336904),
  (12, 19, -4, -14)),
 2: ((0.0094, 0.015201785806984775, 0.00353993619359539, 0.000633616213320412),
  (47, 76, 17, 3)),
 3: ((0.0316, 0.039329505327447466, 0.0224190485924580, 0.0220014030849300),
  (158, 196, 112, 110)),
 4: ((0.0752, 0.07553518218220877, 0.0616472055427024, 0.0668598215597822),
  (376, 377, 308, 334)),
 5: ((0.1366, 0.11486019883481567, 0.113653052913850, 0.123380664175557),
  (683, 574, 568, 616)),
 6: ((0.186, 0.14403265713671407, 0.158862820168806, 0.167839431364716),
  (930, 720, 794, 839)),
 7: ((0.1986, 0.15318274238947757, 0.178462349240122, 0.181590277196423),
  (993, 765, 892, 907)),
 8: ((0.1602, 0.14103336103386102, 0.166474462662479, 0.162631768685471),
  (801, 705, 832, 813)),
 9: ((0.1104, 0.11417899810555138, 0.131342322475809, 0.123385390061926),
  (552, 570, 656, 616)),
 10: ((0.0538, 0.08228992646013068, 0.0882719238974730, 0.0802748700702220),
  (269, 411, 441, 401)),
 11: ((0.0246, 0.05332311999346238, 0.0502170762525970, 0.0448863393840996),
  (123, 266, 251, 224)),
 12: ((0.0082, 0.03132155875841773, 0.0234129221482732, 0.0213134060018062),
  (41, 156, 117, 106)),
 13: ((0.0024, 0.016791973532344077, 0.00797212283913741, 0.00822841571128672),
  (12, 83, 39, 41)),
 14: ((0.0006,
   0.008264412996740285,
   0.000839842377038903,
   0.00220131137645769),
  (3, 41, 4, 11)),
 15: ((0.0, 0.0037526517062144295, -0.00151129408601654, 1.04044204378838e-5),
  (0, 18, -7, 0)),
 17: ((0.0,
   0.0006178821855478986,
   -0.00123182729867254,
   -0.000424421472505167),
  (0, 3, -6, -2))}
```

```python
compare(retrieve_data1olap)
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 2000000,
  'upper_bound': 3000000,
  'interval_length': 100,
  'no_of_checkpoints': 101,
  'contents': ['data',
   'partition',
   'distribution',
   'statistics',
   'comparison - actual, binomial, frei, frei_alt']},
 'data': 
 ...
```

```python
retrieve_data1olap['comparison'][3000000][7]
```
```
((0.193068, 0.15357847695695917, 0.179045618960343, 0.181745123450785),
 (193068, 153578, 179045, 181745))
```

```python
Cnest = list(nest_data1disj['nested_interval_data'].keys())
k = 20
Cnest[k], compare(nest_data1disj)['comparison'][Cnest[k]]
```
```
((2290000, 2710000),
 {0: ((0.0002380952380952381,
    0.0005202764388049725,
    -0.000410030506114741,
    -0.000955584725261755),
   (1, 2, -1, -4)),
  1: ((0.002857142857142857,
    0.004086432288416393,
    -0.000790817080029619,
    -0.00284038956347447),
   (12, 17, -3, -11)),
  2: ((0.009047619047619047,
    0.015887649647662573,
    0.00401593670296977,
    0.00117139026245547),
   (38, 66, 16, 4)),
  3: ((0.033095238095238094,
    0.04076379636644201,
    0.0238225495672815,
    0.0235764897753599),
   (139, 171, 100, 99)),
  4: ((0.08095238095238096,
    0.07764196435230125,
    0.0641435290981889,
    0.0695180641006507),
   (340, 326, 269, 291)),
  5: ((0.14166666666666666,
    0.11708678777086864,
    0.116612880504236,
    0.126289419313985),
   (595, 491, 489, 530)),
  6: ((0.19142857142857142,
    0.14560972933068342,
    0.161107223705815,
    0.169758361134007),
   (804, 611, 676, 712)),
  7: ((0.19047619047619047,
    0.15357847695695917,
    0.179045618960343,
    0.181745123450785),
   (800, 645, 751, 763)),
  8: ((0.15619047619047619,
    0.1402275886955275,
    0.165277417047954,
    0.161157513250706),
   (656, 588, 694, 676)),
  9: ((0.0988095238095238,
    0.11258717762710618,
    0.129019132172140,
    0.121066109017382),
   (415, 472, 541, 508)),
  10: ((0.05476190476190476,
    0.08047119909296573,
    0.0857374941871659,
    0.0779680997004168),
   (230, 337, 360, 327)),
  11: ((0.025952380952380952,
    0.05171308480733388,
    0.0481519901801770,
    0.0431176149383180),
   (109, 217, 202, 181)),
  12: ((0.010952380952380953,
    0.030124465783735684,
    0.0220747010831123,
    0.0202083965034683),
   (46, 126, 92, 84)),
  13: ((0.0030952380952380953,
    0.01601654440308115,
    0.00728530261785767,
    0.00766037549055310),
   (13, 67, 30, 32)),
  14: ((0.0002380952380952381,
    0.007817541352591523,
    0.000586661392106483,
    0.00196894411412870),
   (1, 32, 2, 8)),
  15: ((0.0, 0.003520363646996056, -0.00154213789217459, -5.41122519397797e-5),
   (0, 14, -6, 0)),
  17: ((0.0002380952380952381,
    0.0005700817415597236,
    -0.00117217114204939,
    -0.000408323860716536),
   (1, 2, -4, -1))})
```

```python
Cnest = list(nest_data1olap['nested_interval_data'].keys())
k = 20
Cnest[k], compare(nest_data1olap)['comparison'][Cnest[k]]
```
```
((2290000, 2710000),
 {0: ((8.571428571428571e-05,
    0.0005202764388049725,
    -0.000410030506114741,
    -0.000955584725261755),
   (36, 218, -172, -401)),
  1: ((0.0021142857142857144,
    0.004086432288416393,
    -0.000790817080029619,
    -0.00284038956347447),
   (888, 1716, -332, -1192)),
  2: ((0.010004761904761905,
    0.015887649647662573,
    0.00401593670296977,
    0.00117139026245547),
   (4202, 6672, 1686, 491)),
  3: ((0.033433333333333336,
    0.04076379636644201,
    0.0238225495672815,
    0.0235764897753599),
   (14042, 17120, 10005, 9902)),
  4: ((0.07973809523809523,
    0.07764196435230125,
    0.0641435290981889,
    0.0695180641006507),
   (33490, 32609, 26940, 29197)),
  5: ((0.14002857142857142,
    0.11708678777086864,
    0.116612880504236,
    0.126289419313985),
   (58812, 49176, 48977, 53041)),
  6: ((0.1889047619047619,
    0.14560972933068342,
    0.161107223705815,
    0.169758361134007),
   (79340, 61156, 67665, 71298)),
  7: ((0.19118095238095237,
    0.15357847695695917,
    0.179045618960343,
    0.181745123450785),
   (80296, 64502, 75199, 76332)),
  8: ((0.1585, 0.1402275886955275, 0.165277417047954, 0.161157513250706),
   (66570, 58895, 69416, 67686)),
  9: ((0.10581428571428571,
    0.11258717762710618,
    0.129019132172140,
    0.121066109017382),
   (44442, 47286, 54188, 50847)),
  10: ((0.05559523809523809,
    0.08047119909296573,
    0.0857374941871659,
    0.0779680997004168),
   (23350, 33797, 36009, 32746)),
  11: ((0.02340952380952381,
    0.05171308480733388,
    0.0481519901801770,
    0.0431176149383180),
   (9832, 21719, 20223, 18109)),
  12: ((0.008347619047619048,
    0.030124465783735684,
    0.0220747010831123,
    0.0202083965034683),
   (3506, 12652, 9271, 8487)),
  13: ((0.0022904761904761904,
    0.01601654440308115,
    0.00728530261785767,
    0.00766037549055310),
   (962, 6726, 3059, 3217)),
  14: ((0.0004142857142857143,
    0.007817541352591523,
    0.000586661392106483,
    0.00196894411412870),
   (174, 3283, 246, 826)),
  15: ((0.00010952380952380952,
    0.003520363646996056,
    -0.00154213789217459,
    -5.41122519397797e-5),
   (46, 1478, -647, -22)),
  16: ((1.9047619047619046e-05,
    0.0014689148415780752,
    -0.00165722019733201,
    -0.000489832259683239),
   (8, 616, -696, -205)),
  17: ((9.523809523809523e-06,
    0.0005700817415597236,
    -0.00117217114204939,
    -0.000408323860716536),
   (4, 239, -492, -171)),
  18: ((0.0,
    0.00020646805529753705,
    -0.000682704547063179,
    -0.000244133145442381),
   (0, 86, -286, -102))})
```

<a id='eg2compare'></a>
#### Example 2.

```python
# NB: in the disjoint intervals case, our estimates frei and frei_alt are not intended to be applied.
compare(retrieve_data2disj)
```

```python
compare(retrieve_data2olap)
```

```python
# NB: in the disjoint intervals case, our estimates frei and frei_alt are not intended to be applied.
retrieve_data2disj['comparison'][65558989][5], retrieve_data2olap['comparison'][65558989][5]
```
```
(((0.4166666666666667,
   0.18085653880006008,
   0.205698038678704,
   0.206282102722769),
  (5, 2, 2, 2)),
 ((0.20784313725490197,
   0.18085653880006008,
   0.205698038678704,
   0.206282102722769),
  (212, 184, 209, 210)))
```

```python
Cnest = list(nest_data2disj['nested_interval_data'].keys())
k = 20
Cnest[k], compare(nest_data2disj)['comparison'][Cnest[k]]
```
```
((65639179, 65680759),
 {0: ((0.0021645021645021645,
    0.004269681843779133,
    0.000396634437683792,
    -0.000669024236301658),
   (1, 1, 0, 0)),
  1: ((0.025974025974025976,
    0.02401696037439715,
    0.0113486460010190,
    0.00959716933751003),
   (12, 11, 5, 4)),
  2: ((0.06277056277056277,
    0.06679717105002388,
    0.0498982868802729,
    0.0508133303609056),
   (29, 30, 23, 23)),
  3: ((0.12337662337662338,
    0.12246148027438541,
    0.114938173977106,
    0.120164030159486),
   (57, 56, 53, 55)),
  4: ((0.19696969696969696,
    0.16647107476975395,
    0.176902712246361,
    0.183017055304589),
   (91, 76, 81, 84)),
  5: ((0.2077922077922078,
    0.1789564054008789,
    0.202108676856128,
    0.204297157730229),
   (96, 82, 93, 94)),
  6: ((0.20562770562770563,
    0.15845098396940777,
    0.181299084423810,
    0.178371120810215),
   (95, 73, 83, 82)),
  7: ((0.10606060606060606,
    0.11883823799259051,
    0.131729843957290,
    0.126346200611105),
   (49, 54, 60, 58)),
  8: ((0.047619047619047616,
    0.07705916995839368,
    0.0785587454927965,
    0.0740126390957597),
   (22, 35, 36, 34)),
  9: ((0.017316017316017316,
    0.043880916232043675,
    0.0381723019329632,
    0.0359826421041000),
   (8, 20, 17, 16)),
  10: ((0.0021645021645021645,
    0.02221471384537604,
    0.0143802762415794,
    0.0142318681820293),
   (1, 10, 6, 6)),
  11: ((0.0021645021645021645,
    0.010097597203763622,
    0.00335882539378424,
    0.00421850354773246),
   (1, 4, 1, 1)),
  12: ((0.0, 0.00415474051667502, -0.000423660641089088, 0.000594287863746370),
   (0, 1, 0, 0))})
```

```python
Cnest = list(nest_data2olap['nested_interval_data'].keys())
k = 20
Cnest[k], compare(nest_data2olap)['comparison'][Cnest[k]]
```
```
((65639179, 65680759),
 {0: ((0.0012987012987012987,
    0.004269681843779133,
    0.000396634437683792,
    -0.000669024236301658),
   (54, 177, 16, -27)),
  1: ((0.016378066378066377,
    0.02401696037439715,
    0.0113486460010190,
    0.00959716933751003),
   (681, 998, 471, 399)),
  2: ((0.06955266955266955,
    0.06679717105002388,
    0.0498982868802729,
    0.0508133303609056),
   (2892, 2777, 2074, 2112)),
  3: ((0.1303030303030303,
    0.12246148027438541,
    0.114938173977106,
    0.120164030159486),
   (5418, 5091, 4779, 4996)),
  4: ((0.197017797017797,
    0.16647107476975395,
    0.176902712246361,
    0.183017055304589),
   (8192, 6921, 7355, 7609)),
  5: ((0.2342953342953343,
    0.1789564054008789,
    0.202108676856128,
    0.204297157730229),
   (9742, 7441, 8403, 8494)),
  6: ((0.16998556998557,
    0.15845098396940777,
    0.181299084423810,
    0.178371120810215),
   (7068, 6588, 7538, 7416)),
  7: ((0.10134680134680135,
    0.11883823799259051,
    0.131729843957290,
    0.126346200611105),
   (4214, 4941, 5477, 5253)),
  8: ((0.047546897546897546,
    0.07705916995839368,
    0.0785587454927965,
    0.0740126390957597),
   (1977, 3204, 3266, 3077)),
  9: ((0.023136123136123135,
    0.043880916232043675,
    0.0381723019329632,
    0.0359826421041000),
   (962, 1824, 1587, 1496)),
  10: ((0.0075998075998075995,
    0.02221471384537604,
    0.0143802762415794,
    0.0142318681820293),
   (316, 923, 597, 591)),
  11: ((0.001443001443001443,
    0.010097597203763622,
    0.00335882539378424,
    0.00421850354773246),
   (60, 419, 139, 175)),
  12: ((9.62000962000962e-05,
    0.00415474051667502,
    -0.000423660641089088,
    0.000594287863746370),
   (4, 172, -17, 24)),
  13: ((0.0,
    0.0015580276939568005,
    -0.00108452395089461,
    -0.000305706255221288),
   (0, 64, -45, -12))})
```

<a id='display'></a>
### Display
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Compare](#compare) | ↓ [Plot & animate](#plot) </sup>

Let's make a human-readable display of the data we have gathered.

```python
import pandas as pd

def display(dataset, orient='index', description='on', zeroth_item='show', count='cumulative', comparisons='off', single_cell='true', winners='no show'): 
    # DataFrame orient argument either 'index' or 'columns'.
    # description either 'off' or not (defaults to 'on').
    # zeroth_item either 'no show' or not (defaults to 'show').
    # count either 'partition' or not (defaults to 'cumulative').
    # comparisons either 'absolute', 'probabilities', or not (defaults to 'off').
    # single_cell either 'false' or not (defaults to 'true').
    # winners either 'show' or not (defaults to 'no show').
    if winners == 'show':
        if 'winners' not in dataset.keys():
            return print('Apply the \'winners\' function first.')
        if 'data' in dataset.keys():
            C = list(dataset['data'].keys())
            H = dataset['header']['interval_length']
            interval_type = dataset['header']['interval_type']
            output = {}
            for i in range(1,len(C)):
                if interval_type == 'overlap':
                    output[i] = { 'B - A' : C[i] - C[0], 'A' : C[0], 'B' : C[1],  'H' : H }
                if interval_type == 'disjoint':
                    output[i] = { '(B - A)/H' : (C[i] - C[0])//H, 'A' : C[0], 'B' : C[i],  'H' : H }
		if interval_type == 'prime_start':
                    output[i] = { 'pi(B) - pi(A)' : sum(dataset['nested_interval_data'][C[i]].values()), 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
                for w in dataset['winners'][C[i]]:
                    output[i][w] = dataset['winners'][C[i]][w]   
					
            df = pd.DataFrame.from_dict(output, orient=orient)
            return df           
        if 'nested_interval_data' in dataset.keys():
            C = list(dataset['nested_interval_data'].keys())
            H = dataset['header']['interval_length']
            interval_type = dataset['header']['interval_type']
            output = {}
            for i in range(len(C)):
                if interval_type == 'overlap':
                    output[i] = { 'B - A' : C[i][1] - C[i][0], 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
                if interval_type == 'disjoint':
                    output[i] = { '(B - A)/H' : (C[i][1] - C[i][0])//H, 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
		if interval_type == 'prime_start':
                    output[i] = { 'pi(B) - pi(A)' : sum(dataset['nested_interval_data'][C[i]].values()), 'A' : C[i][0], 'B' : C[i][1],  'H' : H }            
                for w in dataset['winners'][C[i]]:
                    output[i][w] = dataset['winners'][C[i]][w]   
            df = pd.DataFrame.from_dict(output, orient=orient)
            return df        
    else:
        if 'data' in dataset.keys():
            if comparisons == 'absolute' or comparisons == 'probabilities':
                if comparisons == 'absolute':
                    index = 1
                if comparisons == 'probabilities':
                    index = 0
                if 'comparison' not in dataset.keys():
                    return print('First compare the data to something with the compare function.')
                if count == 'partition':
                    return print('We only compare cumulative (non-partitioned) data.')
                C = list(dataset['comparison'].keys())
                C.sort()
                output = { C[0] : { m : 0 for m in dataset['comparison'][C[0]].keys()} }
                for c in C[1:]:
                    output[c] = {}
                    for m in dataset['comparison'][c].keys():
                        output[c][m] = dataset['comparison'][c][m][index]

                df = pd.DataFrame.from_dict(output, orient=orient)

            else:
                if count == 'partition':
                    if 'partition' not in dataset.keys():
                        return print('First partition the data.')
                    datakey = 'partition'        
                else:
                    if 'data' not in dataset.keys():
                        return print('First unpartition the data.')
                    datakey = 'data' 
                C = list(dataset[datakey].keys())
                C.sort()
                output = {}
                # In the case of disjoint intervals, we can display 'prime tallies' for each checkpoint.
                # (Gives the total number of primes from C[0] to C[k] in the cumulative count case,
                # or from C[k-1] to C[k] in the partial count case).
                # In the case of displaying the partitioned data (count 'partial' i.e. non-cumulative), 
                # we can show totals at the end of each row/column (depending on the orientation), giving the 
                # total number of intervals between A and B that contain m primes.
                # (In the cumulative count case, the totals are just the last row/column anyway.)
                for c in C:
                    output[c] = {}
                    for m in dataset[datakey][c].keys():
                        output[c][m] = dataset[datakey][c][m]        
                if dataset['header']['interval_type'] == 'disjoint':      
                    for c in C:
                        output[c]['prime_tally'] = {}
                        tally = sum([m*dataset[datakey][c][m] for m in dataset[datakey][c].keys()])
                        output[c]['prime_tally'] = tally        
                if count == 'partition':
                    output['totals'] = {}
                    for m in dataset[datakey][C[-1]].keys():
                        output['totals'][m] = sum([dataset[datakey][c][m] for c in C])
                    if dataset['header']['interval_type'] == 'disjoint':
                        #output['totals']['prime_tally'] = sum([m*output['totals'][m] for m in dataset[datakey][C[-1]].keys()])
                        output['totals']['prime_tally'] = sum([output[c]['prime_tally'] for c in C]) # should be the same as above

                df = pd.DataFrame.from_dict(output, orient=orient)    

            if description == 'off':
                if zeroth_item == 'no show':
                    if orient == 'columns':
                        A = dataset['header']['lower_bound']
                        return df.loc[:, df.columns!=A]
                    else: 
                        return df.tail(-1)
                else:
                    return df            
            else:
                interval_type = dataset['header']['interval_type']
                if interval_type == 'overlap':
                    word = 'overlapping'
                if interval_type == 'disjoint':
                    word = 'disjoint'
                if interval_type == 'prime_start':
                    word = 'left endpoint prime'
                A = dataset['header']['lower_bound']
                B = dataset['header']['upper_bound']
                H = dataset['header']['interval_length']
                if count == 'partition':
                    counts = 'non-cumulative'
                else:
                    counts = 'cumulative'
                text = f'Interval type: {word}. Lower bound: {A}. Upper bound: {B}. Interval length: {H}. Partial counts: {counts}.'        
                if comparisons == 'absolute' or comparisons == 'probabilities':
                    text = text + 'In tuple (a,b,c,d), a is actual data, b is Binomial prediction, c is frei prediction, and d is frei_alt prediction.'
                if zeroth_item == 'no show':
                    if orient == 'columns':
                        return df.loc[:, df.columns!=A].style.set_caption(text)
                    else:
                        return df.tail(-1).style.set_caption(text)
                else:
                    return df.style.set_caption(text)
        if 'nested_interval_data' in dataset.keys():
            if comparisons == 'absolute' or comparisons == 'probabilities':
                if comparisons == 'absolute':
                    index = 1
                if comparisons == 'probabilities':
                    index = 0
                if 'comparison' not in dataset.keys():
                    return print('First compare the data to something with the compare function.')        
            C = list(dataset['nested_interval_data'].keys())
            H = dataset['header']['interval_length']
            interval_type = dataset['header']['interval_type']
            M = list(dataset['nested_interval_data'][C[-1]].keys())         
            output = {}
            for i in range(len(C)):
                if interval_type == 'overlap':
                    output[i] = { 'B - A' : C[i][1] - C[i][0], 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
                if interval_type == 'disjoint':
                    output[i] = { '(B - A)/H' : (C[i][1] - C[i][0])//H, 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
                if interval_type == 'prime_start':
                    output[i] = { 'pi(B) - pi(A)' : sum(dataset['nested_interval_data'][C[i]].values()), 'A' : C[i][0], 'B' : C[i][1],  'H' : H }
                if not(comparisons == 'absolute' or comparisons == 'probabilities'):
                    for m in M:
                        output[i][m] = dataset['nested_interval_data'][C[i]][m]
                    if interval_type == 'disjoint':
                        tally = sum([m*output[i][m] for m in M])
                        output[i]['prime tally'] = tally
                else:                
                    if single_cell=='true':
                        for m in M:
                            output[i][m] = dataset['comparison'][C[i]][m][index]                        
                    else:
                        Mexpand = []
                        for m in M:
                            Mexpand.extend([m,f'B{m}', f'F{m}', f'F*{m}'])
                        j = 0
                        while j < len(Mexpand):
                            m = Mexpand[j]
                            B = Mexpand[j + 1]
                            F = Mexpand[j + 2]
                            Falt = Mexpand[j + 3]
                            output[i][m] = dataset['comparison'][C[i]][m][index][0]
                            output[i][B] = dataset['comparison'][C[i]][m][index][1]
                            output[i][F] = dataset['comparison'][C[i]][m][index][2]
                            output[i][Falt] = dataset['comparison'][C[i]][m][index][3]
                            j += 4

            df = pd.DataFrame.from_dict(output, orient=orient)
            return df
```

<a id='eg1display'></a>
#### Example 1.

```python
display(retrieve_data1disj)
```
```
Interval type: disjoint. Lower bound: 2000000. Upper bound: 3000000. Interval length: 100. Partial counts: cumulative.
 	0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	17	prime_tally
2000000	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
2010000	0	0	0	3	7	13	25	10	13	16	10	2	0	1	0	0	0	705
2020000	0	0	0	6	15	23	45	32	31	30	12	3	2	1	0	0	0	1395
2030000	0	0	2	11	24	32	58	55	46	41	20	8	2	1	0	0	0	2088
2040000	0	0	4	13	28	47	74	80	68	45	26	11	2	2	0	0	0	2778
...
```

```python
display(retrieve_data1olap, zeroth_item = 'no show', orient='columns', count='partition')
```
```
Interval type: overlap. Lower bound: 2000000. Upper bound: 3000000. Interval length: 100. Partial counts: non-cumulative.
 	2010000	2020000	2030000	2040000	2050000	2060000	2070000	2080000	2090000	2100000	2110000	2120000	2130000	2140000	2150000	2160000	2170000	2180000	2190000	2200000	2210000	2220000	2230000	2240000	2250000	2260000	2270000	2280000	2290000	2300000	2310000	2320000	2330000	2340000	2350000	2360000	2370000	2380000	2390000	2400000	2410000	2420000	2430000	2440000	2450000	2460000	2470000	2480000	2490000	2500000	2510000	2520000	2530000	2540000	2550000	2560000	2570000	2580000	2590000	2600000	2610000	2620000	2630000	2640000	2650000	2660000	2670000	2680000	2690000	2700000	2710000	2720000	2730000	2740000	2750000	2760000	2770000	2780000	2790000	2800000	2810000	2820000	2830000	2840000	2850000	2860000	2870000	2880000	2890000	2900000	2910000	2920000	2930000	2940000	2950000	2960000	2970000	2980000	2990000	3000000	totals
0	0	48	0	0	0	0	0	0	0	0	0	0	6	0	0	0	0	0	0	0	0	0	0	8	6	0	0	0	0	0	2	8	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	10	0	4	0	12	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	6	0	0	20	0	0	0	0	0	0	0	0	0	0	130
1	0	12	0	28	38	12	0	0	18	28	12	12	8	14	0	0	20	16	22	78	4	0	0	26	100	0	0	0	52	34	46	86	8	26	38	26	0	0	4	4	52	2	6	0	0	0	2	16	116	2	0	30	50	14	28	0	18	0	0	32	10	50	74	14	8	0	0	66	0	16	10	0	8	18	4	54	10	0	42	0	0	0	10	26	80	0	14	20	0	32	34	38	20	10	24	14	0	0	10	26	1882
2	38	20	122	100	108	62	70	138	80	190	82	110	26	84	58	40	130	112	70	196	102	48	102	60	102	76	18	30	126	86	98	84	70	104	184	152	54	78	104	136	146	56	40	56	50	106	80	156	148	104	76	104	142	218	80	34	62	32	68	88	90	160	184	94	138	14	56	174	98	74	124	30	128	156	90	164	128	42	96	52	38	82	164	52	190	58	70	112	48	126	86	132	200	58	136	142	14	104	126	162	9688
3	314	136	438	160	242	304	422	342	272	260	344	424	288	194	232	208	376	200	242	338	318	346	376	394	216	282	302	254	340	244	276	306	238	348	404	324	200	658	334	320	434	292	286	430	186	310	422	480	256	334	406	446	304	518	328	216	156	204	300	206	340	520	428	240	352	118	398	438	408	264	370	352	486	334	188	282	412	358	346	386	238	362	458	210	538	256	262	432	274	310	326	358	424	410	522	462	428	350	328	326	33024
...
```

```python
display(nest_data1disj)
```
```
(B - A)/H	A	B	H	0	1	2	3	4	5	...	8	9	10	11	12	13	14	15	17	prime tally
0	200	2490000	2510000	100	0	0	4	6	11	27	...	33	21	5	2	3	2	0	0	0	1350
1	400	2480000	2520000	100	0	2	8	9	35	49	...	64	43	17	9	4	4	0	0	0	2712
2	600	2470000	2530000	100	0	3	9	20	52	67	...	101	67	30	14	5	4	0	0	0	4081
3	800	2460000	2540000	100	0	3	11	29	69	99	...	121	80	48	20	8	4	0	0	0	5421
4	1000	2450000	2550000	100	0	3	12	36	84	126	...	154	101	57	23	9	6	0	0	0	6778
...
```

```python
display(nest_data1olap,comparisons='absolute')
```
```
B - A	A	B	H	0	1	2	3	4	5	...	9	10	11	12	13	14	15	16	17	18
0	20000	2490000	2510000	100	(0, 10, -8, -19)	(2, 81, -15, -56)	(180, 317, 80, 23)	(740, 815, 476, 471)	(1350, 1552, 1282, 1390)	(2728, 2341, 2332, 2525)	...	(2132, 2251, 2580, 2421)	(880, 1609, 1714, 1559)	(334, 1034, 963, 862)	(126, 602, 441, 404)	(66, 320, 145, 153)	(6, 156, 11, 39)	(0, 70, -30, -1)	(0, 29, -33, -9)	(0, 11, -23, -8)	(0, 4, -13, -4)
1	40000	2480000	2520000	100	(0, 20, -16, -38)	(148, 163, -31, -113)	(432, 635, 160, 46)	(1442, 1630, 952, 943)	(2962, 3105, 2565, 2780)	(5350, 4683, 4664, 5051)	...	(4398, 4503, 5160, 4842)	(2152, 3218, 3429, 3118)	(840, 2068, 1926, 1724)	(314, 1204, 882, 808)	(90, 640, 291, 306)	(6, 312, 23, 78)	(0, 140, -61, -2)	(0, 58, -66, -19)	(0, 22, -46, -16)	(0, 8, -27, -9)
```

<a id='eg2display'></a>
#### Example 2.

```python
display(retrieve_data2disj, count='partition', description='off')
```
```
	0	1	2	3	4	5	6	7	8	9	10	11	prime_tally
65557969	0	0	0	0	0	0	0	0	0	0	0	0	0
65558989	0	1	1	2	1	5	1	0	0	0	1	0	54
65560009	0	0	0	0	5	5	1	1	0	0	0	0	58
65561029	0	0	2	2	2	3	2	1	0	0	0	0	52
65562049	0	0	1	0	3	4	1	1	2	0	0	0	63
...
```

```python
display(retrieve_data2olap, comparisons='absolute')
```
```

Interval type: overlap. Lower bound: 65557969. Upper bound: 65761969. Interval length: 85. Partial counts: cumulative.In tuple (a,b,c,d), a is actual data, b is Binomial prediction, c is frei prediction, and d is frei_alt prediction.
 	0	1	2	3	4	5	6	7	8	9	10	11	12	13
65557969	0	0	0	0	0	0	0	0	0	0	0	0	0	0
65558989	(0, 5, 0, 0)	(40, 31, 16, 14)	(128, 82, 65, 66)	(177, 142, 138, 144)	(195, 182, 197, 202)	(212, 184, 209, 210)	(98, 153, 174, 170)	(74, 108, 117, 112)	(51, 66, 64, 60)	(35, 35, 28, 26)	(10, 16, 8, 9)	(0, 7, 1, 2)	(0, 2, 0, 0)	(0, 0, -1, 0
```

```python
display(retrieve_data2olap, comparisons='probabilities')
```
```
Interval type: overlap. Lower bound: 65557969. Upper bound: 65761969. Interval length: 85. Partial counts: cumulative.In tuple (a,b,c,d), a is actual data, b is Binomial prediction, c is frei prediction, and d is frei_alt prediction.
 	0	1	2	3	4	5	6	7	8	9	10	11	12	13
65557969	0	0	0	0	0	0	0	0	0	0	0	0	0	0
65558989	(0.0, 0.005778701344443381, 0.000932176816482944, -0.000274753949948496)	(0.0392156862745098, 0.030702319208985536, 0.0162667183428929, 0.0147023853235282)	(0.12549019607843137, 0.08060138050813222, 0.0638839571106127, 0.0656644335066840)	(0.17352941176470588, 0.1393866964422863, 0.135503517530820, 0.141330163736975)	(0.19117647058823528, 0.17860647258590573, 0.193582650718642, 0.198980306616352)	(0.20784313725490197, 0.18085653880006008, 0.205698038678704, 0.206282102722769)	(0.09607843137254903, 0.15072835483128574, 0.171435216785510, 0.167359058604099)	(0.07254901960784314, 0.10632760152069547, 0.115269055868157, 0.109909302065536)	(0.05, 0.06479964704118212, 0.0630520825794851, 0.0593502753929423)	(0.03431372549019608, 0.03465316155125581, 0.0275328435090937, 0.0262669539437814)	(0.00980392156862745, 0.016461843276652573, 0.00876751064257008, 0.00917009947316973)	(0.0, 0.007015668311226328, 0.00114426938529099, 0.00214810123188904)	(0.0, 0.002704216940090859, -0.000943991229544109, -1.60931581594569e-5)	(0.0, 0.0009491679036882558, -0.00100071604751012, -0.000376006860067233)
...
```

```python
display(nest_data2disj,comparisons='probabilities')
```
```
(B - A)/H	A	B	H	0	1	2	3	4	5	6	7	8	9	10	11	12
0	22	65658979	65660959	90	(0.0, 0.004269681843779133, 0.0003966344376837...	(0.045454545454545456, 0.02401696037439715, 0....	(0.045454545454545456, 0.06679717105002388, 0....	(0.09090909090909091, 0.12246148027438541, 0.1...	(0.18181818181818182, 0.16647107476975395, 0.1...	(0.18181818181818182, 0.1789564054008789, 0.20...	(0.18181818181818182, 0.15845098396940777, 0.1...	(0.2727272727272727, 0.11883823799259051, 0.13...	(0.0, 0.07705916995839368, 0.0785587454927965,...	(0.0, 0.043880916232043675, 0.0381723019329632...	(0.0, 0.02221471384537604, 0.0143802762415794,...	(0.0, 0.010097597203763622, 0.0033588253937842...	(0.0, 0.00415474051667502, -0.0004236606410890...
1	44	65657989	65661949	90	(0.0, 0.004269681843779133, 0.0003966344376837...	(0.06818181818181818, 0.02401696037439715, 0.0...	(0.06818181818181818, 0.06679717105002388, 0.0...	(0.11363636363636363, 0.12246148027438541, 0.1...	(0.20454545454545456, 0.16647107476975395, 0.1...	(0.20454545454545456, 0.1789564054008789, 0.20...	(0.18181818181818182, 0.15845098396940777, 0.1...	(0.13636363636363635, 0.11883823799259051, 0.1...	(0.022727272727272728, 0.07705916995839368, 0....	(0.0, 0.043880916232043675, 0.0381723019329632...	(0.0, 0.02221471384537604, 0.0143802762415794,...	(0.0, 0.010097597203763622, 0.0033588253937842...	(0.0, 0.00415474051667502, -0.0004236606410890..
```

```python
display(nest_data2olap).tail()
```
```
B - A	A	B	H	0	1	2	3	4	5	6	7	8	9	10	11	12	13
96	192060	65563939	65755999	90	408	3120	12408	26114	38316	40653	32506	21257	10560	4770	1528	358	54	8
97	194040	65562949	65756989	90	414	3200	12470	26294	38664	41097	33038	21447	10674	4794	1528	358	54	8
98	196020	65561959	65757979	90	414	3200	12517	26544	39044	41519	33382	21728	10880	4834	1538	358	54	8
99	198000	65560969	65758969	90	414	3212	12660	26885	39450	41874	33758	21938	10985	4860	1544	358	54	8
100	199980	65559979	65759959	90	414	3212	12769	27089	40018	42404	34116	22080	11054	4860	1544	358	54	8
```

<a id='eg3display'></a>
#### Example 3 (table from Gauss's Nachlass).

```python
# We can now re-create Goldschmidt's table in one line, in about one second.

display(partition(intervals(list(range(2*10**6,3*10**6 + 1, 10**5)), 100, 'disjoint')), orient='columns', zeroth_item='no show', count='partition')
```
```

Interval type: disjoint. Lower bound: 2000000. Upper bound: 3000000. Interval length: 100. Partial counts: non-cumulative.
 	2100000	2200000	2300000	2400000	2500000	2600000	2700000	2800000	2900000	3000000	totals
0	0	0	0	0	0	0	1	0	0	0	1
1	3	2	2	4	1	3	4	2	2	2	25
2	10	9	9	10	9	5	10	7	15	13	97
3	32	27	29	33	37	35	28	43	30	43	337
4	69	70	73	86	78	88	70	93	84	65	776
5	119	145	138	135	146	136	159	137	141	152	1408
6	198	183	179	177	193	193	195	195	179	189	1881
7	203	201	205	194	190	179	201	188	222	212	1995
8	158	167	168	157	151	172	141	145	131	135	1525
9	114	110	113	113	102	88	96	86	110	103	1035
10	63	52	44	54	56	57	54	68	53	58	559
11	21	18	30	29	25	25	22	24	18	15	227
12	8	9	10	7	7	13	17	9	7	11	98
13	2	4	0	1	5	6	1	2	6	1	28
14	0	3	0	0	0	0	1	0	2	0	6
15	0	0	0	0	0	0	0	0	0	1	1
17	0	0	0	0	0	0	0	1	0	0	1
prime_tally	6872	6857	6849	6791	6770	6808	6765	6717	6747	6707	67883
```

<a id='plot'></a>
### Plot & animate
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Display](#display) | ↓ [Worked example](#worked) </sup>

<a id='eg1plot'></a>
#### Example 1.

```python
import numpy as np
import sympy
import matplotlib.pyplot as plt 
from matplotlib import animation  
from matplotlib import rc  

X = retrieve_data1olap

interval_type = X['header']['interval_type']
A = X['header']['lower_bound']
H = X['header']['interval_length']
C = list(X['distribution'].keys())

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle('Primes in intervals')

hor_axis = list(X['distribution'][C[-1]].keys())
y_min, y_max = 0, 0
for c in C:
    for m in X['distribution'][c].keys():
        if y_max < X['distribution'][c][m]:
            y_max = X['distribution'][c][m]
    
def plot(cp):
    ax.clear()

    mu = X['statistics'][cp]['mean']
    sigma = X['statistics'][cp]['var']
    med = X['statistics'][cp]['med']
    if med == int(med):
        med = int(med)
    modes = X['statistics'][cp]['mode']
    
    # Bounds for the plot, and horizontal axis tick marks. 
    ax.set(xlim=(hor_axis[0], hor_axis[-1]), ylim=(0, np.ceil(100*y_max)/100 ))

    # The data and histogram
    ver_axis = list(X['distribution'][cp].values())
    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  

    # Predictions for comparison
    B = cp
    N = (A + B)/2
    p = 1/(np.log(N) - 1)
    p_alt = 1/np.log(N)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    #ax.plot(x, pois_pmf(H,x,H*p), 'r--',zorder=3.5, label=r'$\mathrm{Pois}(\lambda)$')
    #ax.plot(x, norm.pdf(x,H*p,np.sqrt(H*p*(1 - p))), 'g--',zorder=3.5)
    ax.plot(x, binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
    if interval_type == 'overlap':
        ax.plot(x, frei(H,x,H*p), 'g--',zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
        #ax.plot(x, frei_alt(H,x,H*p_alt), 'b--', zorder=3.5, label=r'$\mathrm{F^*}(H,m,\lambda^*)$')
    
    # Overlay information
    if interval_type == 'overlap':
        ax.text(0.70,0.15,fr'$X = \pi(a + H) - \pi(a)$, ' 
                + fr'$A < a \leq B$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + fr'$A = {A}$' + '\n\n' 
                + fr'$B = {B}$' + '\n\n' 
                + fr'$N = (A + B)/2$' + '\n\n' 
                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
    if interval_type == 'disjoint':
        ax.text(0.72,0.1,fr'$X = \pi(a + H) - \pi(a)$, ' 
                + fr'$a = A + kH$' + '\n\n' 
                + fr'$0 \leq k \leq (B - A)/H$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + fr'$A = {A}$' + '\n\n' 
                + fr'$B = {B}$' + '\n\n' 
                + fr'$N = (A + B)/2$' + '\n\n' 
                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)

    
    # Formating/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r'$m$ (number of primes in an interval)')
    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else. 
    ax.grid(True,zorder=0,alpha=0.7)   
    
# Generate the animation
X_anim = animation.FuncAnimation(fig, plot, frames=C[1:], interval=100, blit=False, repeat=False)

# This is supposed to remedy the blurry axis ticks/labels. 
plt.rcParams['savefig.facecolor'] = 'white'

plot(C[-1])
plt.show()
```
![SegmentLocal](images/README/2_to_3_million_H_100.png)

```python
# Save a video of the animation.
from IPython.display import HTML

HTML(X_anim.to_html5_video())
```

<a id='eg2plot'></a>
#### Example 2.

```python
X = retrieve_data2olap

interval_type = X['header']['interval_type']
A = X['header']['lower_bound']
H = X['header']['interval_length']
C = list(X['distribution'].keys())

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle('Primes in intervals')

hor_axis = list(X['distribution'][C[-1]].keys())
y_min, y_max = 0, 0
for c in C[1:]:
    for m in X['distribution'][c].keys():
        if y_max < X['distribution'][c][m]:
            y_max = X['distribution'][c][m]
    
def plot(cp):
    ax.clear()

    mu = X['statistics'][cp]['mean']
    sigma = X['statistics'][cp]['var']
    med = X['statistics'][cp]['med']
    if med == int(med):
        med = int(med)
    modes = X['statistics'][cp]['mode']
    
    # Bounds for the plot, and horizontal axis tick marks. 
    ax.set(xlim=(hor_axis[0]-0.5, hor_axis[-1]+0.5), ylim=(0,np.ceil(100*y_max)/100 ))

    # The data and histogram
    ver_axis = list(X['distribution'][cp].values())
    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  

    # Predictions for comparison
    B = cp
    N = (A + B)/2
    p = 1/(np.log(N) - 1)
    p_alt = 1/np.log(N)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    #ax.plot(x, pois_pmf(H,x,H*p), 'r--',zorder=3.5, label=r'$\mathrm{Pois}(\lambda)$')
    #ax.plot(x, norm.pdf(x,H*p,np.sqrt(H*p*(1 - p))), 'g--',zorder=3.5)
    ax.plot(x, binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
    if interval_type == 'overlap':
        #ax.plot(x, frei_alt(H,x,H*p_alt), 'b--',zorder=3.5, label=r'$\mathrm{F^*}(H,m,\lambda^*)$')
        ax.plot(x, frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
    
    # Overlay information
    if interval_type == 'overlap':
        if B != C[-1]:
            ax.text(0.70,0.18,fr'$X = \pi(a + H) - \pi(a)$, ' 
                    +  fr'$A < a \leq B$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$A = {A}$' + '\n\n' 
                    + fr'$B = {B}$' + '\n\n' 
                    + fr'$N = (A + B)/2$' + '\n\n' 
                    + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' 
                    + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
        if B == C[-1]:
            ax.text(0.74,0.18,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                    + fr'$N - M < a \leq N + M$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$N = [\exp(18)]$' + '\n\n' 
                    + fr'$M = {B - int(np.exp(18))}$' + '\n\n' 
                    + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' 
                    + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)            
    if interval_type == 'disjoint':
        ax.text(0.72,0.1,fr'$X = \pi(a + H) - \pi(a)$, ' 
                + fr'$a = A + kH$' + '\n\n' 
                + fr'$0 \leq k \leq (B - A)/H$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + fr'$A = {A}$' + '\n\n' 
                + fr'$B = {B}$' + '\n\n' 
                + fr'$N = (A + B)/2$' + '\n\n' 
                + fr'$\lambda = H/\log N = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)

    
    # Formating/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r'$m$ (number of primes in an interval)')
    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else. 
    ax.grid(True,zorder=0,alpha=0.7)   
    
# Generate the animation
X_anim = animation.FuncAnimation(fig, plot, frames=C[1:], interval=100, blit=False, repeat=False)

# This is supposed to remedy the blurry axis ticks/labels. 
plt.rcParams['savefig.facecolor'] = 'white'

plot(C[-1])
plt.show()
```
![SegmentLocal](images/README/exp18_H_85.png)

```python
# Save a video of the animation.
from IPython.display import HTML

HTML(X_anim.to_html5_video())
```

<a id='worked'></a>
### Worked examples
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Plot & animate](#plot) | ↓ [Extensions](#extensions) </sup>

<a id='eg4worked'></a>
### Example 4

```python
# In this example, we'll try our alternative estimate F*(H,m,lambda*), lambda* = 1/log N.
N = int(np.exp(20))
HH = [20,30,40,50,60,70,80,90,100,200]
step = 10**3
A = N - 10**5
B = N + 10**5
C = list(range(A,B+1,step))
```
```python
exp20 = { H : {} for H in HH}
start = timer()
for H in HH:
    exp20[H] = intervals(C,H,interval_type='overlap')
end = timer()
end - start 
```
``
3146.539329699939
```

```python
for H in HH:
    save(exp20[H])
```
```python
for H in [20,30,40,50,70,80,200]:
    retrieve(H)['header']
```
```
Found 1 dataset corresponding to interval of length 20 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 20, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 30 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 30, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 40 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 40, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 50 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 50, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 70 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 70, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 80 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 80, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 200 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 200, 'no_of_checkpoints': 201, 'contents': ['data']}
```

```python
retrieve(60)[1]['header']
```
```
Found 2 datasets corresponding to interval of length 60 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 2669017, 'upper_bound': 3869017, 'interval_length': 60, 'no_of_checkpoints': 1201, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 60, 'no_of_checkpoints': 201, 'contents': ['data']}

{'interval_type': 'overlap',
 'lower_bound': 485065195,
 'upper_bound': 485265195,
 'interval_length': 60,
 'no_of_checkpoints': 201,
 'contents': ['data']}
```

```python
getexp20 = {}
for H in [20,30,40,50,70,80,200]:
    getexp20[H] = retrieve(H)
```
```
Found 1 dataset corresponding to interval of length 20 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 20, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 30 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 30, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 40 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 40, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 50 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 50, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 70 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 70, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 80 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 80, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 1 dataset corresponding to interval of length 200 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 200, 'no_of_checkpoints': 201, 'contents': ['data']}
```

```python
for H in [60, 90, 100]:
    getexp20[H] = retrieve(H)[1]
```
```
Found 2 datasets corresponding to interval of length 60 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 2669017, 'upper_bound': 3869017, 'interval_length': 60, 'no_of_checkpoints': 1201, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 60, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 2 datasets corresponding to interval of length 90 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 65559979, 'upper_bound': 65759959, 'interval_length': 90, 'no_of_checkpoints': 203, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 90, 'no_of_checkpoints': 201, 'contents': ['data']}

Found 2 datasets corresponding to interval of length 100 (overlap intervals).

 [0] 'header' : {'interval_type': 'overlap', 'lower_bound': 2000000, 'upper_bound': 3000000, 'interval_length': 100, 'no_of_checkpoints': 101, 'contents': ['data']}


 [1] 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 100, 'no_of_checkpoints': 201, 'contents': ['data']}
```

```python
for H in HH:
    analyze(getexp20[H])
```
```python
for H in HH:
    compare(getexp20[H])
```

```python
H = 90 # 20,30,40,50,70,80,200
X = getexp20[H]

interval_type = X['header']['interval_type']
A = X['header']['lower_bound']
H = X['header']['interval_length']
C = list(X['distribution'].keys())

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle('Primes in intervals')

hor_axis = list(X['distribution'][C[-1]].keys())
y_min, y_max = 0, 0
for c in C[1:]:
    for m in X['distribution'][c].keys():
        if y_max < X['distribution'][c][m]:
            y_max = X['distribution'][c][m]
    
def plot(cp):
    ax.clear()

    mu = X['statistics'][cp]['mean']
    sigma = X['statistics'][cp]['var']
    med = X['statistics'][cp]['med']
    if med == int(med):
        med = int(med)
    modes = X['statistics'][cp]['mode']
    
    # Bounds for the plot, and horizontal axis tick marks. 
    ax.set(xlim=(hor_axis[0]-0.5, hor_axis[-1]+0.5), ylim=(0,np.ceil(100*y_max)/100 ))

    # The data and histogram
    ver_axis = list(X['distribution'][cp].values())
    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  

    # Predictions for comparison
    B = cp
    N = (A + B)/2
    p_alt = 1/np.log(N)
    p = 1/(np.log(N) - 1)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    #ax.plot(x, pois_pmf(H,x,H*p), 'r--',zorder=3.5, label=r'$\mathrm{Pois}(\lambda)$')
    #ax.plot(x, norm.pdf(x,H*p,np.sqrt(H*p*(1 - p))), 'g--',zorder=3.5)
    ax.plot(x, binom_pmf(H,x,p_alt), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda^*/H)$')
    if interval_type == 'overlap':
        ax.plot(x, frei_alt(H,x,H*p_alt), 'b--',zorder=3.5, label=r'$\mathrm{F^*}(H,m,\lambda^*)$')
        #ax.plot(x, frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
    
    # Overlay information
    if interval_type == 'overlap':
        if B != C[-1]:
            ax.text(0.70,0.18,fr'$X = \pi(a + H) - \pi(a)$, ' 
                    +  fr'$A < a \leq B$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$A = {A}$' + '\n\n' 
                    + fr'$B = {B}$' + '\n\n' 
                    + fr'$N = (A + B)/2$' + '\n\n' 
                    + fr'$\lambda^* = H/\log N = {H*p_alt:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' 
                    + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
        if B == C[-1]:
            ax.text(0.74,0.18,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                    + fr'$N - M < a \leq N + M$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$N = [\exp(20)]$' + '\n\n' 
                    + fr'$M = {int(B - N)}$' + '\n\n' 
                    + fr'$\lambda^* = H/\log N = {H*p_alt:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' 
                    + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)            
    if interval_type == 'disjoint':
        ax.text(0.72,0.1,fr'$X = \pi(a + H) - \pi(a)$, ' 
                + fr'$a = A + kH$' + '\n\n' 
                + fr'$0 \leq k \leq (B - A)/H$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + fr'$A = {A}$' + '\n\n' 
                + fr'$B = {B}$' + '\n\n' 
                + fr'$N = (A + B)/2$' + '\n\n' 
                + fr'$\lambda = H/\log N = {H*p_alt:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)

    
    # Formating/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r'$m$ (number of primes in an interval)')
    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else. 
    ax.grid(True,zorder=0,alpha=0.7)   
    
# Generate the animation
X_anim = animation.FuncAnimation(fig, plot, frames=C[1:], interval=100, blit=False, repeat=False)

# This is supposed to remedy the blurry axis ticks/labels. 
plt.rcParams['savefig.facecolor'] = 'white'

plot(C[-1])
plt.show()
```

![SegmentLocal](images/README/exp20alt_H_90.png)

```python
# Save a video of the animation.
from IPython.display import HTML

HTML(X_anim.to_html5_video())
```

<a id='eg5worked'></a>
### Example 5

```python
N = int(np.exp(20))
M = 10**5
H = 76
A = N - M
B = N + M
step = 10**3
C = list(range(A, B + 1, step))
N_exp20_H_76 = intervals(C, H)
```
```python
N_exp20_H_76 = retrieve(76)
```
```
Found 1 dataset corresponding to interval of length 76 (overlap intervals).

 'header' : {'interval_type': 'overlap', 'lower_bound': 485065195, 'upper_bound': 485265195, 'interval_length': 76, 'no_of_checkpoints': 201, 'contents': ['data']}
```

```python
analyze(N_exp20_H_76)
```
```
{'header': {'interval_type': 'overlap',
  'lower_bound': 485065195,
  'upper_bound': 485265195,
  'interval_length': 76,
  'no_of_checkpoints': 201,
  'contents': ['data', 'distribution', 'statistics']},
 'data': {485065195: {0: 0,
 ...
```

```python
X = N_exp20_H_76

interval_type = X['header']['interval_type']
A = X['header']['lower_bound']
H = X['header']['interval_length']
C = list(X['distribution'].keys())

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle('Primes in intervals')

hor_axis = list(X['distribution'][C[-1]].keys())
y_min, y_max = 0, 0
for c in C[1:]:
    for m in X['distribution'][c].keys():
        if y_max < X['distribution'][c][m]:
            y_max = X['distribution'][c][m]
    
def plot(cp):
    ax.clear()

    mu = X['statistics'][cp]['mean']
    sigma = X['statistics'][cp]['var']
    med = X['statistics'][cp]['med']
    if med == int(med):
        med = int(med)
    modes = X['statistics'][cp]['mode']
    
    # Bounds for the plot, and horizontal axis tick marks. 
    ax.set(xlim=(hor_axis[0]-0.5, hor_axis[-1]+0.5), ylim=(0,np.ceil(100*y_max)/100 ))

    # The data and histogram
    ver_axis = list(X['distribution'][cp].values())
    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  

    # Predictions for comparison
    B = cp
    N = (A + B)/2    
    p = 1/(np.log(N) - 1)
    p_alt = 1/np.log(N)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    #ax.plot(x, pois_pmf(H,x,H*p), 'r--',zorder=3.5, label=r'$\mathrm{Pois}(\lambda)$')
    #ax.plot(x, norm.pdf(x,H*p,np.sqrt(H*p*(1 - p))), 'g--',zorder=3.5)
    ax.plot(x, binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
    if interval_type == 'overlap':
        #ax.plot(x, frei_alt(H,x,H*p_alt), 'b--',zorder=3.5, label=r'$\mathrm{F^*}(H,m,\lambda^*)$')
        ax.plot(x, frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
    
    # Overlay information
    if interval_type == 'overlap':
        if B != C[-1]:
            ax.text(0.70,0.18,fr'$X = \pi(a + H) - \pi(a)$, ' 
                    +  fr'$A < a \leq B$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$A = {A}$' + '\n\n' 
                    + fr'$B = {B}$' + '\n\n' 
                    + fr'$N = (A + B)/2$' + '\n\n' 
                    + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' 
                    + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
        if B == C[-1]:
            ax.text(0.74,0.18,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                    + fr'$N - M < a \leq N + M$' + '\n\n' 
                    + fr'$H = {H}$' + '\n\n' 
                    + fr'$N = [\exp(20)]$' + '\n\n' 
                    + fr'$M = {int(B - N)}$' + '\n\n' 
                    + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                    + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                    + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                    + fr'median : ${med}$' + '\n\n' + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)            
    if interval_type == 'disjoint':
        ax.text(0.72,0.1,fr'$X = \pi(a + H) - \pi(a)$, ' + fr'$a = A + kH$' + '\n\n' 
                + fr'$0 \leq k \leq (B - A)/H$' + '\n\n' + fr'$H = {H}$' + '\n\n' 
                + fr'$A = {A}$' + '\n\n' + fr'$B = {B}$' + '\n\n' + fr'$N = (A + B)/2$' + '\n\n' 
                + fr'$\lambda = H/\log N = {H*p:.5f}$' + '\n\n' + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' + fr'median : ${med}$' + '\n\n' + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)

    
    # Formating/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r'$m$ (number of primes in an interval)')
    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else. 
    ax.grid(True,zorder=0,alpha=0.7)   
    
# Generate the animation
X_anim = animation.FuncAnimation(fig, plot, frames=C[1:], interval=100, blit=False, repeat=False)

# This is supposed to remedy the blurry axis ticks/labels. 
plt.rcParams['savefig.facecolor'] = 'white'

plot(C[-1])
plt.show()
```
![SegmentLocal](images/README/exp20_H_76.png)

```python
# Save a video of the animation.
from IPython.display import HTML

HTML(X_anim.to_html5_video())
```

```python
# from matplotlib.animation import PillowWriter

# Save the animation as an animated GIF

# f = r"xxx\images\exp20_H_76.gif" 
# X_anim.save(f, dpi=100, writer='imagemagick', extra_args=['-loop','1'], fps=10)

# extra_args=['-loop','1'] for no looping, '0' for looping.

# f = r"xxx\images\exp20_H_76_loop.gif" 
# X_anim.save(f, dpi=100, writer='imagemagick', extra_args=['-loop','0'], fps=10)

# MovieWriter stderr:
# magick.exe: unable to extend cache '-': No space left on device @ error/cache.c/OpenPixelCache/3914.
```

<a id='eg6worked'></a>
#### Example 6 (nested intervals)

```python
N = int(np.exp(21))
K = 199
k = (K - 1)//2 # = 99
Delta = 10**3 
D = []
for j in range(k + 1):
    D.extend([N - (j + 1)*Delta, N + (j + 1)*Delta])
C = sorted(D)
HH = list(range(20,141,10))
EXP21 = {H : {} for H in HH}
```

```python
for H in HH:
    EXP21[H] = intervals(C, H)
# Takes a few hours...
```

```python
for H in HH:
    save(EXP21[H])
```

```python
NEXP21 = {}
for H in HH:
    NEXP21[H] = nest(EXP21[H])
for H in HH:
    analyze(NEXP21[H])
```

```python
# HH = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
X = NEXP21[HH[4]]

interval_type = X['header']['interval_type']
A = X['header']['lower_bound']
H = X['header']['interval_length']
C = list(X['distribution'].keys())

plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle('Primes in intervals')

hor_axis = list(X['distribution'][C[-1]].keys())
y_min, y_max = 0, 0
for c in C:
    for m in X['distribution'][c].keys():
        if y_max < X['distribution'][c][m]:
            y_max = X['distribution'][c][m]
    
def plot(c):
    ax.clear()

    mu = X['statistics'][c]['mean']
    sigma = X['statistics'][c]['var']
    med = X['statistics'][c]['med']
    if med == int(med):
        med = int(med)
    modes = X['statistics'][c]['mode']
    
    # Bounds for the plot, and horizontal axis tick marks. 
    ax.set(xlim=(hor_axis[0]-0.5, hor_axis[-1]+0.5), ylim=(0,np.ceil(1000*y_max)/1000 ))

    # The data and histogram
    ver_axis = list(X['distribution'][c].values())
    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  

    # Predictions for comparison
    A = c[0]
    B = c[1]
    N = (A + B)//2
    exponent= str(int(np.log(N)) + 1)
    M = N - A
    k = M//10**3
    p = 1/(np.log(N) - 1)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    ax.plot(x, binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
    ax.plot(x, frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
    
    # Overlay information
    if B != C[-1][1]:
        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                +  fr'$N - M < a \leq N + M$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + r'$N = [e^{21}]$' + '\n\n' 
                + fr'$M = 10^3k$, $k = {k}$' + '\n\n' 
                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
    if B == C[-1][1]:
        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                +  fr'$N - M < a \leq N + M$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + r'$N = [e^{21}]$' + '\n\n' 
                + fr'$M = 10^5$' + '\n\n' 
                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
    # Formating/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r'$m$ (number of primes in an interval)')
    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else. 
    ax.grid(True,zorder=0,alpha=0.7)   
    
# Generate the animation
X_anim = animation.FuncAnimation(fig, plot, frames=C, interval=100, blit=False, repeat=False)

# This is supposed to remedy the blurry axis ticks/labels. 
plt.rcParams['savefig.facecolor'] = 'white'

plot(C[-1])
plt.show()
```
```python
# Save a video of the animation.
from IPython.display import HTML

HTML(X_anim.to_html5_video())
```

![SegmentLocal](images/README/N_exp21_H_60.gif)

<a id='extensions'></a>
### Extensions
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Worked example](#worked) | ↓ [References](#references) </sup>

To get some insight into what's going on here, it may help to know exactly which intervals contain a given number of primes. We can add a few lines to our code in order to output a list of $a$ for which $(a, a + H]$ contains exactly $m$ primes, for $m$ in a given list of interest.

```python
def overlap_extension(A,B,H,M):
    P = postponed_sieve()
    Q = postponed_sieve()
    output = { m : 0 for m in range(H + 1) } 
    show_me = {m : [] for m in M}
    a = A + 1 
    p, q = next(P), next(Q) 
    while p < a + 1:
        p, q = next(P), next(Q) 
    m = 0 
    while q < a + H + 1: 
        m += 1
        q = next(Q) 
    while p < B + 1:
        if m in M:
            show_me[m].append(a)
        output[m] += 1    
        b, c = p - a, q - (a + H) 
        if m in M:
            show_me[m].extend([x for x in range(a + 1,a + min(b,c))])
        output[m] = output[m] + min(b,c) - 1
        if b == c:
            a = p
            p = next(P)
        if b < c:
            a, m = p, m - 1
            p = next(P)
        if c < b:
            a, m = a + c, m + 1
        while q < a + H + 1:
            q = next(Q)
    while a < B + 1: 
        if m in M:
            show_me[m].append(a)
        output[m] += 1
        b, c = p - a, q - (a + H) 
        if a + min(b,c) > B: 
            if m in M:
                show_me[m].extend([x for x in range(a + 1,B + 1)])
            output[m] = output[m] + B - a
            break
        else:  
            if m in M:
                show_me[m].extend([x for x in range(a + 1,a + c)])
            output[m] = output[m] + c - 1
            a, m = a + c, m + 1
            while q < a + H + 1:
                q = next(Q)
    output = { m : output[m] for m in output.keys() if output[m] != 0}
    return show_me, output
```

```python
overlap_extension(1000,2000,50,[3,11])
```
```
({3: [1307, 1308, 1309, 1310, 1321, 1322, 1327, 1328, 1329, 1330],
  11: [1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278]},
 {3: 10, 4: 94, 5: 138, 6: 202, 7: 216, 8: 178, 9: 136, 10: 18, 11: 8})
```

<a id='references'></a>
### References
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Extensions](#extensions) </sup>

[1] Freiberg, T. "[A note on the distribution of primes in intervals](http://dx.doi.org/10.1007/978-3-319-92777-0_2)." pp 23–44 in _Irregularities in the distribution of prime numbers_. Eds. J. Pintz and M. Th. Rassias. Springer, 2018.

[2] Gallagher, P. X. "[On the distribution of primes in short intervals](http://dx.doi.org/10.1112/S0025579300016442)." _Mathematika_ 23(1):4–9, 1976.

[3] Montgomery, H. L. and K. Soundararajan. "[Primes in short intervals.](http://dx.doi.org/10.1007/s00220-004-1222-4)." _Commun. Math. Phys._ 252(1-3):589–617, 2004.

[4] Tschinkel, Y. "[About the cover: on the distribution of primes–Gauss' tables](https://doi.org/10.1090/S0273-0979-05-01096-7)." Bull. Amer. Math. Soc. 43(1)89–91, 2006.
