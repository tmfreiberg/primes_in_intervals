## Biases in the distribution of primes in intervals

<a id='tldr'></a>
### TL;DR
<sup>Jump to: ↓↓ [Contents](#contents) | ↓ [Introduction](#introduction) </sup>

The histogram shows the distribution of primes in intervals. Cramér's random model leads to a prediction for this distribution, shown in orange. We have another prediction, shown in green. In the data we've looked at, as here, our prediction seems to fare better than the one based on Cramér's model. 

As suggested in [[1]](#references), our prediction is based on the Hardy-Littlewood prime tuples conjecture, inclusion-exclusion, and a precise estimate, due to Montgomery and Soundararajan [[3]](#references), for a certain average involving the singular series of the prime tuples conjecture. 

![SegmentLocal](images/exp20/exp20_H_76.png)

The prediction of Cramér's random model (orange) is, with $\lambda/H = 1/(\log N - 1)$ the "probability" that an integer close to $N$ is prime,

$$\mathrm{Binom}(H,\lambda/H) =  \frac{e^{-\lambda}\lambda^m}{m!}\bigg[1 + \frac{Q_1(\lambda,m)}{H} + \frac{Q_2(\lambda,m)}{H^2} + \cdots\bigg],$$
where each $Q_j(\lambda,m)$ is a polynomial in $\lambda$ and $m$, and in particular, 

$$Q_1(\lambda,m) = \frac{m - (m - \lambda)^2}{2}.$$

Our prediction (green) is 

$$F(H,m,\lambda) = \frac{e^{-\lambda}\lambda^m}{m!}\left[1 + \frac{\log H + (\log 2\pi + \gamma - 1)}{H}Q_1(\lambda,m) \right],$$

in agreement with Cramér's model only as a first-order approximation. The secondary term in our prediction is more in line with our observation that the distribution of the numerical data is more "pinched up" around the center: there is more of a _bias_ towards the mean $\lambda$ than is suggested by the Binomial distribution.

<a id='introduction'></a>
### Introduction
<sup>Jump to: ↓ [Contents](#contents) | ↑ [TL;DR](#tldr)</sup>

Excerpt from letter from Gauss to his student Johann Franz Encke, December 24, 1849 (see [[4]](#references)).

_In 1811, the appearance of Chernau's cribrum gave me much pleasure and I have frequently (since I lack the patience
for a continuous count) spent an idle quarter of an hour to count another chiliad here and there; although I eventually gave it up without quite getting through a million. Only some time later did I make use of the diligence of **Goldschmidt** to fill some of the remaining gaps in the first million and to continue the computationa ccording to Burkhardt’s tables. Thus (for many years now) the first three million have been counted and checked against the integral. A small excerpt follows..._

![SegmentLocal](images/nachlass.jpg)

![SegmentLocal](images/goldschmidt_table_plot.png)

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
..........[A single function](#single_function)<br>
..........[To do](#to_do)<br>
[Raw data](#raw_data)<br> .......... [Example 1](#eg1generate) | [Example 2](#eg2generate)<br>
[Save](#save)<br> .......... [Example 1](#eg1save) | [Example 2](#eg2save)<br>
[Retrieve](#retrieve)<br> .......... [Example 1](#eg1retrieve) | [Example 2](#eg2retrieve)<br>
[Narrow](#narrow)<br> .......... [Example 1](#eg1narrow) | [Example 2](#eg2narrow)<br>
[Partition](#partition)<br> .......... [Example 1](#eg1partition) | [Example 2](#eg2partition)<br>
[Analyze](#analyze)<br> .......... [Example 1](#eg1analyze) | [Example 2](#eg2analyze)<br>
[Compare](#compare)<br> .......... [Example 1](#eg1compare) | [Example 2](#eg2compare)<br>
[Display](#display)<br> .......... [Example 1](#eg1display) | [Example 2](#eg2display) | [Example 3](#eg3display) (table from Gauss's _Nachlass_)<br>
[Plot & animate](#plot)<br> .......... [Example 1](#eg1plot) | [Example 2](#eg2plot)<br>
[Worked examples](#worked)<br> .......... [Example 4](#eg4worked) | [Example 5](#eg5worked)<br>
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
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Overlapping intervals](#overlapping) | ↓ [A single function](#single_function) </sup>

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

<a id='prime_endpoint'></a>
#### Prime left endpoint intervals
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Overlapping intervals, with checkpoints](#overlapping_checkpoints) | ↓ [A single function](#single_function) </sup>

See [to do](#to_do)

<a id='single_function'></a>
#### A single function
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [Overlapping intervals, with checkpoints](#overlapping_checkpoints) | ↓ [To do](#to_do) </sup>

```python
def intervals(C,H,interval_type='overlap'): 
    if interval_type == 'disjoint':
        return disjoint_cp(C,H)
    if interval_type == 'overlap': 
        return overlap_cp(C,H)
```

<a id='to_do'></a>
#### To do
<sup>Jump to: ↑↑ [Contents](#contents) | ↑ [A single function](#single_function) | ↓ [Raw data](#raw_data) </sup>

* A version of the intervals function for ```interval_type == 'prime_start'```, meaning that we consider only intervals of the form $(a, a + H]$, where $a$ is _prime_.

* If we do a computation that takes a long time, and then want to extend the calculation, we'd like to be able to pick up where we left off. Suppose we compute ```intervals([0,N],H)``` where ```N``` is very large, and then we'd like to compute ```intervals([N,2N], H)``` or ```intervals([0,2N], H)```. At the moment, we'd have to start from scratch. What we'd like to do is save the state of our ```intervals``` function, particularly the prime generators, and then just keep going.

* In a similar vein, another thing we could do is input various values for the interval length ```H```, say a list ```[H_1,H_2,...,H_k]```, and have a function return ```intervals(C,H_i)``` for each ```H_i```, without simply computing ```intervals(C,H_i)``` $k$ times.

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

