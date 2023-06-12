```python
import primes_in_intervals as pii
```
**Example 1.** Tables from Gauss's _Nachlass_. Create data for primes in disjoint intervals of length $100$ (starting at multiples of $100$), up to $10^7$. Checkpoints every $10^5$.

By the bye, an interval of the form $[100k, 100k + 100)$ is referred to as a "Centade" in Gauss's _Nachlass_. Analogous to the term Decade (a 10-year period from years '0 to '9), a Centade is a 100-year period from '00 to '99. 

```python
C = list(range(0,10*10**6 + 1, 10**5))
H = 100
nachlass = pii.intervals(C,H,'disjoint')
```

Gauss and Goldschmidt summarize their data in tables of primes up to $1$ million, between $1$ and $2$ million, and so on. Let's emulate that.

```python
NACHLASS = { }
for i in range(1,11):
    NACHLASS[i] = pii.extract(nachlass, [(i - 1)*10**6, i*10**6] , option='narrow')
    pii.partition(NACHLASS[i])
```

Let's display one of these tables: the one for primes between $2$ and $3$ million.

```python
NACHLASS3df = pii.display(NACHLASS[3], count='partition', orient='columns')
pii.dfi.export(NACHLASS3df, 'NACHLASS3df.png')
```
![SegmentLocal](images/examples/NACHLASS3df.png)

Here's the original: Gauss/Goldschmidt were only short by $21$ primes in the end!

![SegmentLocal](images/examples/nachlass.jpg)

**Example 2.** Let's look at a series of nested intervals centred around $N = [e^{17}] = 24,154,952$. We take the density of primes close to $N$ as $1/(\log N - 1)$, which is $1/15.999999968802452\ldots$, virtually $1/16$. We'll get data for intervals of length $64, 68, 72, 76, 80$. 
```python
import numpy as np
N = int(np.exp(17))
HH = [64, 68, 72, 76, 80]
C = list(range(N - 10**4,N + 10**4 + 1, 10**2))
EXP17 = {}
for H in HH:
    EXP17[H] = pii.intervals(C, H, 'overlap')
```
Right now we have data for the intervals $(N  - 10^4, N - 10^4 + 100k]$, $k = 1,\ldots,100$. Let's reconfigure the data to consider nested intervals $(N - 100k, N + 100k]$, $k = 1,\ldots,100$, all centred around $N$. We'll analyze the data (get the distributions and statistics) while we're at it.

```python
EXP17NEST = {}
for H in HH:
    EXP17NEST[H] = pii.nest(EXP17[H])
    pii.analyze(EXP17NEST[H])
```

Let's display what we have for $H = 76$ for instance.

```python
EXP17_76_NESTtable = pii.display(EXP17NEST[76])
pii.dfi.export(EXP17_76_NESTtable, 'EXP17_76_NESTtable.png')
```
![SegmentLocal](images/examples/EXP17_76_NESTtable.png)

Let's compare the data (for $H = 76$) to three predictions: the Binomial, our prediction, and the alternate version of our prediction (with the density of primes around $N$ taken to be $1/\log N$). Specifically, we $\lambda = H/(\log N - 1)$ (and assuming $\lambda \asymp 1$), the naive prediction based purely on Cramér's model is

$$\mathrm{Binom}(H,\lambda/H) =  \frac{e^{-\lambda}\lambda^m}{m!}\bigg[1 + \frac{Q_1(\lambda,m)}{H} + \frac{Q_2(\lambda,m)}{H^2} + \cdots\bigg],$$
where each $Q_j(\lambda,m)$ is a polynomial in $\lambda$ and $m$, and in particular, 

$$Q_1(\lambda,m) = \frac{m - (m - \lambda)^2}{2}.$$

Our prediction is 

$$F(H,m,\lambda) = \frac{e^{-\lambda}\lambda^m}{m!}\left[1 + \frac{Q_1(\lambda,m)}{H}\left(\log H + (\log 2\pi + \gamma - 1)\right) \right].$$

Our alternative prediction is, with $\lambda^\* = 1/\log N$, 

$$F^\*(H,m,\lambda^\*) = \frac{e^{-\lambda^\*}(\lambda^\*)^m}{m!}\left[1 + \frac{\lambda^\*}{H}(m - \lambda^\*) + \frac{\log H + (\log 2\pi + \gamma - 1)}{H}Q_1(m,\lambda^\*) \right].$$

The table below shows tuples $(a,b,c,d)$, where $a$ is the actual number, $b$ is the Binomial-based prediction, $c$ is our prediction and $d$ is our alternative prediction. It would be nice to be able to conjecture something about a tertiary term!

We'll just show the last five rows of the table as it's a bit long and hard to read.

```python
pii.compare(EXP17NEST[76])
EXP17_76_NESTcompare = pii.display(EXP17NEST[76], comparisons='absolute').tail(5)
pii.dfi.export(EXP17_76_NESTcompare, 'EXP17_76_NESTcompare_tail.png')
```
![SegmentLocal](images/examples/EXP17_76_NESTcompare_tail_5.png)

We can perhaps work on the formatting of such tables.

Now, we want to know which prediction is the "best", and this is hard to see by glancing at the above table. By "best" we mean gives the smallest sum-of-squared-error over $m$ (number of primes in an interval). We're mainly interested in comparing $F$ and $F^\*$. Comparing $F^\*$ to the Binomial is not really apples-to-apples because the Binomial we are using takes the probability of finding a prime around $N$ as being $\lambda/H = 1/(\log N - 1)$, rather than $\lambda^\*/H = 1/\log N$, as in $F^\*$.

We'll use our 'winners' function to determine the best predictions for each interval. For each prediction, this function also gives us the $m$ for which that prediction gives a smaller error than the others. 

```python
pii.winners(EXP17NEST[76])
EXP17_76_NESTwinners = pii.display(EXP17NEST[76], winners='show')
pii.dfi.export(EXP17_76_NESTwinners, 'EXP17_76_NESTwinners.png')
```

![SegmentLocal](images/examples/EXP17_76_NESTwinners.png)

Finally, let's make an animated plot, with one frame for each of the intervals considered.

```python
import matplotlib.pyplot as plt # for plotting distributions
from matplotlib import animation # for animating sequences of plots
from matplotlib import rc # to help with the animation
from IPython.display import HTML # to save animations
from matplotlib.animation import PillowWriter # to save animations as a gif

# HH = [64, 68, 72, 76, 80]
X = EXP17NEST[HH[0]]

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
    k = M//10**2
    p = 1/(np.log(N) - 1)
    x = np.linspace(hor_axis[0],hor_axis[-1],100)
    ax.plot(x, pii.binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
    ax.plot(x, pii.frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
    
    # Overlay information
    if B != C[-1][1]:
        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                +  fr'$N - M < a \leq N + M$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + r'$N = [e^{17}]$' + '\n\n' 
                + fr'$M = 10^2k$, $k = {k}$' + '\n\n' 
                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
                + fr'median : ${med}$' + '\n\n' 
                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
    if B == C[-1][1]:
        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
                +  fr'$N - M < a \leq N + M$' + '\n\n' 
                + fr'$H = {H}$' + '\n\n' 
                + r'$N = [e^{17}]$' + '\n\n' 
                + fr'$M = 10^4$' + '\n\n' 
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

The final frame looks like this:

![SegmentLocal](images/examples/EXP17_76_NESTplot.png)

Save the animation...

```python
HTML(X_anim.to_html5_video())
```

![SegmentLocal](images/examples/EXP17_76_NESTanim.gif)









