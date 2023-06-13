# PRIMES IN INTERVALS BY TRISTAN FREIBERG

# FOR AN EXPOSITION WITH EXAMPLES SEE https://github.com/tmfreiberg/primes_in_intervals/blob/main/README.md

# LIBRARIES/PACKAGES/MODULES

from itertools import count                     # used in the postponed_sieve prime generator
import sqlite3                                  # to save and retrieve data from a database
import pandas as pd                             # to display tables of data
import dataframe_image as dfi                   # to save images of dataframes
from timeit import default_timer as timer       # to see how long certain computations take
import numpy as np                              # for square-roots, exponentials, logs, etc.
from scipy.special import binom as binom        # binom(x,y) = Gamma(x + 1)/[Gamma(y + 1)Gamma(x - y + 1)]
from scipy.special import gamma as gamma        # Gamma function
# from scipy.stats import norm                   # don't need this at present
import sympy                                    # for the Euler-Mascheroni constant, EulerGamma, in the constant 1 - gamma - log(2*pi) from Montgomery-Soundararajan
import matplotlib.pyplot as plt                 # for plotting distributions
from matplotlib import animation                # for animating sequences of plots
from matplotlib import rc                       # to help with the animation
from IPython.display import HTML                # to save animations
from matplotlib.animation import PillowWriter   # to save animations as a gif

# PRIME GENERATOR

# Prime generator found here: https://stackoverflow.com/questions/2211990/how-to-implement-an-efficient-infinite-generator-of-prime-numbers-in-python
# This code was posted by Will Ness. See above URL for further information about who contributed what, and discussion of complexity.

                                         # ideone.com/aVndFM
def postponed_sieve():                   # postponed sieve, by Will Ness      
    yield 2; yield 3; yield 5; yield 7;  # original code David Eppstein, 
    sieve = {}                           #   Alex Martelli, ActiveState Recipe 2002
    ps = postponed_sieve()               # a separate base Primes Supply:
    p = next(ps) and next(ps)            # (3) a Prime to add to dict
    q = p*p                              # (9) its sQuare 
    for r in count(9,2):                 # the Candidate
        if r in sieve:                  # r's a multiple of some base prime
            s = sieve.pop(r)            #     i.e. a composite ; or
        elif r < q:  
             yield r                    # a prime
             continue              
        else:   # (r==q):               # or the next base prime's square:
            s=count(q+2*p,2*p)          #    (9+6, by 6 : 15,21,27,33,...)
            p=next(ps)                  #    (5)
            q=p*p                       #    (25)
        for m in s:                     # the next multiple 
            if m not in sieve:          # no duplicates
                break
        sieve[m] = s                    # original test entry: ideone.com/WFv4f

        
# DISJOINT INTERVALS WITH CHECKPOINTS

# ANCILLARY FUNCTION: REMOVE/ADD ZERO-ITEMS FROM/TO META-DICTIONARY

def zeros(meta_dictionary, pad='yes'):
    # pad option is either explicitly 'no' or not (defaults to 'yes').
    output = {}
    if pad == 'no':
        for k in meta_dictionary.keys():
            output[k] = { m : meta_dictionary[k][m] for m in meta_dictionary[k].keys() if meta_dictionary[k][m] != 0}
        return output
    # if pad option is 'yes' or unspecified or anything other than the string 'no'...
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

# DISJOINT INTERVALS WITH CHECKPOINTS

# Input checkpoint list C and interval length H.
# C is re-defined if necessary to take the form [C[0], C[1],..., C[K]], where C[k] are all distinct and C[k] = C[0] mod H for each k.
# Output is "meta-dictionary" with 'header' item and 'data' item.
# 'header' item contains information to help identify the data.
# 'data' item's values are dictionaries of the form { C[k] : { 0 : g(0), 1 : g(1), ...} } where g(m) is the number of intervals (a, a + H] that contain exactly m primes, with a ranging over ([C[0], C[0] + H, C[0] + 2*H, ..., C[k]]

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

# OVERLAPPING INTERVALS

# Input checkpoint list C and interval length H.
# Output is "meta-dictionary" with 'header' item and 'data' item.
# 'header' item contains information to help identify the data.
# 'data' item's values are dictionaries of the form { C[k] : { 0 : h(0), 1 : h(1), ...} } where h(m) is the number of intervals (a, a + H] that contain exactly m primes, with a ranging over ([C[0],C[k]]

def overlap_cp(C,H):
    output = { 'header' : {'interval_type' : 'overlap', 'lower_bound' : C[0], 'upper_bound' : C[-1], 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : []} }
    C.sort()
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

# A SINGLE FUNCTION

def intervals(C,H,interval_type='overlap'):
    # interval_type is either 'disjoint' or not (defaults to 'overlap' unless 'disjoint' is explicitly given).
    if interval_type == 'disjoint':
        return disjoint_cp(C,H)
    # if interval_type is 'overlap' or not given or is anything string other than 'disjoint'
    if interval_type == 'overlap': 
        return overlap_cp(C,H)

 # ANY INTERVALS: A MORE GENERAL AND ELEGANT (BUT SLOWER) FUNCTION

# This is the core function. We need to make a "checkpoint" version and update all the related functions (display, etc.).

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
        while a + H <= min(b, N + H):     
            if a + H == b:  
                m += 1
                Blist.append(b)
                b = next(B)
            for i in range(len(Blist)):
                if Blist[i] <= a:
                    m += -1 
                    Blist[i] = 'x'
            while 'x' in Blist:
                Blist.remove('x')
            output[m] += 1   
            a = next(A)
    output = { m : output[m] for m in range(H + 1) if output[m] != 0}
    return output

      
# SAVE

# We have a database called primes_in_intervals_db.
# Tables therein include one for disjoint intervals (disjoint_raw) and one for overlapping intervals (overlap_raw).
# First column shall be A (lower bound), second column B (upper bound), third column H (interval length), where we consider intervals of the form (a, a + H] for a in (A,B] (a in an arithmetic progression mod H in the disjoint case).
# The first three columns (A,B,H) will constitute each table's primary key.
# The next max_primes columns shall contain the number of such intervals with m primes, where m = 0,1,...,max_primes will easily cover any situation we will be interested in, where

max_primes = 100 # can change this and alter tables in future if need be

# Generate the string 'm0 int, m1 int, m2 int, ... '
cols = ''
for i in range(max_primes + 1):
    cols = cols + 'm' + f'{i}' + ' int, '

conn = sqlite3.connect('primes_in_intervals_db')
conn.execute('CREATE TABLE IF NOT EXISTS disjoint_raw (lower_bound int, upper_bound int, interval_length int,' + cols + 'PRIMARY KEY(lower_bound, upper_bound, interval_length))')
conn.execute('CREATE TABLE IF NOT EXISTS overlap_raw (lower_bound int, upper_bound int, interval_length int,' + cols + 'PRIMARY KEY(lower_bound, upper_bound, interval_length))')
conn.commit()
conn.close()

# So, A, B, H, m0, m1, ..., m[max_primes] are columns 0, 1, 2, 3, ..., max_primes + 3, respectively: mi is column i + 3.

# The save function stores data in the appropriate database table (as determined by the 'interval_type' item in the data's 'header' dictionary).

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
    conn.commit()
    conn.close()

# RETRIEVE

# First, we define a function that shows an entire table in our database.

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


# Now we define a function that takes H as an input and reconstructs the original dictionary(ies) we created that correspond to interval length H.

# Retrieve data from our database table. Recall that we have the row C[0], C[k], H, g(0), g(1), ... , g(max_primes) in our disjoint_raw table (similarly in our overlap_raw table). 
# We want to reconstruct the dictionary
# {'header' : {'interval_type' : 'disjoint/overlap', 'lower_bound' : A, 'upper_bound' : B, 'interval_length' : H, 'no_of_checkpoints' : len(C), 'contents' : ['data']}, 'data' : { C[0] : {m : g(m), ...}, C[1] : { m : g(m), ... } }  }

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

# NARROW OR FILTER

# Input a dataset and a list C.
# Output a NEW dataset with info about primes in intervals (a, a + H] with a in (A,B] (option = 'narrow'), or with checkpoints common to newC and the current checkpoints (option = 'filter').

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

# PARTITION

# Input a dataset with checkpoints A = C_0, C_1,..., C_k = B.
# MODIFY the dataset with info about primes in intervals (a, a + H] with a in (C_{i-1},C_i] for i = 1,...,k.
# Put this info in a new item 'partition' : { ... }, and append 'partition' to the 'contents' of the 'header'.

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

# We can reverse this as well...

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

# NESTED INTERVALS

# Input a dataset with data corresponding to checkpoints [C_0,...,C_K], where C_0 < C_1 < ... < C_K. 
# Output a NEW dataset with data corresponding to the intervals (assuming K = 2k + 1 is odd)(C_k, C_{k + 1}], (C_{k-1}, C_{k + 2}], ..., (C_0, C_K].
# Note that each interval is contained in the next, so these are "nested" intervals.
# If the C's form an arithmetic progression, then each of the nested intervals share a common midpoint.

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

# ANALYZE

# ANCILLARY FUNCTIONS

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
# The output is a dictionary, whose first item is itself a dictionary, whose keys are the same as the input dictionary, and for which each key's value is the _proportion_ of occurrences (_relative_ frequency) of the key among the data.
# The other items in the output dictionary are mean, variance, median, mode, etc., of the original data.

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

# Now for the main function.
# Input a dataset containing a 'data' item, and MODIFY the dataset by adding a new 'distribution' item and a new 'statistics' item, using our ancillary dictionary_statistics function.

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


# COMPARE

# The three functions used to make predictions about the distribution of primes in intervals are binom_pmf, frei, and frei_alt.

def binom_pmf(H,m, p):
    return binom(H,m)*(p**m)*(1 - p)**(H - m)

MS = 1 - sympy.EulerGamma.evalf() - np.log(2*(np.pi)) # "Montgomery-Soundararajan" constant
def frei(H,m,t):
    Q_2 = ((m - t)**2 - m)/2
    return np.exp(-t)*(t**m/gamma(m + 1))*(1 - ((np.log(H) - MS)/(H))*Q_2)

def frei_alt(H,m,t):
    Q_1 = m - t
    Q_2 = ((m - t)**2 - m)/2
    return np.exp(-t)*(t**m/gamma(m + 1))*(1 + (t/H)*Q_1 - ((np.log(H) - MS)/(H))*Q_2)

# And now the compare function.
# Input a dataset and MODIFY it by adding a 'comparison' dictionary that is analogous to the 'data' dictionary, but instead of the inner-most values being g(m) or h(m) (number of intervals containing m primes), they are tuples consiting of the actual values and three predictions.

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
            for m in dataset['data'][c].keys():
                binom_prob = binom_pmf(H,m,p)
                frei_prob = frei(H,m,H*p)
                frei_alt_prob = frei_alt(H,m,H*p_alt)
                binom_pred = int(binom_prob*multiplier) # what dataset['data'][c][m] should be according to Cramer's model
                frei_pred = int(frei_prob*multiplier) # what dataset['data'][c][m] should be up to second-order approximation, at least around the centre of the distribution, according to me
                frei_alt_pred = int(frei_alt_prob*multiplier) # the alternative estimate
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
            for m in dataset['nested_interval_data'][c].keys():
                binom_prob = binom_pmf(H,m,p)
                frei_prob = frei(H,m,H*p)
                frei_alt_prob = frei_alt(H,m,H*p_alt)
                binom_pred = int(binom_prob*multiplier) # what dataset['data'][c][m] should be according to Cramer's model
                frei_pred = int(frei_prob*multiplier) # what dataset['data'][c][m] should be up to second-order approximation, at least around the centre of the distribution, according to me
                frei_alt_pred = int(frei_alt_prob*multiplier) # the alternative estimate
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
    
# FIND THE BEST PREDICTION

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

# DISPLAY

# Input a dataset and some options.
# Output a dataframe that displays the data.
# A different presentation for the 'nested intervals' data.

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
                A = dataset['header']['lower_bound']
                B = dataset['header']['upper_bound']
                H = dataset['header']['interval_length']
                if count == 'partition':
                    counts = 'non-cumulative'
                else:
                    counts = 'cumulative'
                text = f'Interval type: {interval_type}. Lower bound: {A}. Upper bound: {B}. Interval length: {H}. Partial counts: {counts}.'        
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
        
        
# PLOT AND ANIMATE

###

# WORKED EXAMPLE

# Here is an example of the whole process (nested intervals).

##N = int(np.exp(21))
##K = 199
##k = (K - 1)//2 # = 99
##Delta = 10**3 
##D = []
##for j in range(k + 1):
##    D.extend([N - (j + 1)*Delta, N + (j + 1)*Delta])
##C = sorted(D)
##HH = list(range(20,141,10))
##EXP21 = {H : {} for H in HH}
##
##for H in HH:
##    EXP21[H] = intervals(C, H)
##
##for H in HH:
##    save(EXP21[H])
##
##NEXP21 = {}
##for H in HH:
##    NEXP21[H] = nest(EXP21[H])
##for H in HH:
##    analyze(NEXP21[H])
##
### HH = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
##X = NEXP21[HH[4]]
##
##interval_type = X['header']['interval_type']
##A = X['header']['lower_bound']
##H = X['header']['interval_length']
##C = list(X['distribution'].keys())
##
##plt.rcParams.update({'font.size': 22})
##
##fig, ax = plt.subplots(figsize=(22, 11))
##fig.suptitle('Primes in intervals')
##
##hor_axis = list(X['distribution'][C[-1]].keys())
##y_min, y_max = 0, 0
##for c in C:
##    for m in X['distribution'][c].keys():
##        if y_max < X['distribution'][c][m]:
##            y_max = X['distribution'][c][m]
##    
##def plot(c):
##    ax.clear()
##
##    mu = X['statistics'][c]['mean']
##    sigma = X['statistics'][c]['var']
##    med = X['statistics'][c]['med']
##    if med == int(med):
##        med = int(med)
##    modes = X['statistics'][c]['mode']
##    
##    # Bounds for the plot, and horizontal axis tick marks. 
##    ax.set(xlim=(hor_axis[0]-0.5, hor_axis[-1]+0.5), ylim=(0,np.ceil(1000*y_max)/1000 ))
##
##    # The data and histogram
##    ver_axis = list(X['distribution'][c].values())
##    ax.bar(hor_axis, ver_axis, color='#e0249a', zorder=2.5, alpha=0.3, label=r'$\mathrm{Prob}(X = m)$')
##    ax.plot(hor_axis, ver_axis, 'o', color='red', zorder=2.5)  
##
##    # Predictions for comparison
##    A = c[0]
##    B = c[1]
##    N = (A + B)//2
##    exponent= str(int(np.log(N)) + 1)
##    M = N - A
##    k = M//10**3
##    p = 1/(np.log(N) - 1)
##    x = np.linspace(hor_axis[0],hor_axis[-1],100)
##    ax.plot(x, binom_pmf(H,x,p), '--', color='orange', zorder=3.5, label=r'$\mathrm{Binom}(H,\lambda/H)$')
##    ax.plot(x, frei(H,x,H*p), '--', color='green', zorder=3.5, label=r'$\mathrm{F}(H,m,\lambda)$')
##    
##    # Overlay information
##    if B != C[-1][1]:
##        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
##                +  fr'$N - M < a \leq N + M$' + '\n\n' 
##                + fr'$H = {H}$' + '\n\n' 
##                + r'$N = [e^{21}]$' + '\n\n' 
##                + fr'$M = 10^3k$, $k = {k}$' + '\n\n' 
##                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
##                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
##                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
##                + fr'median : ${med}$' + '\n\n' 
##                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
##    if B == C[-1][1]:
##        ax.text(0.75,0.15,fr'$X = \pi(a + H) - \pi(a)$' + '\n\n' 
##                +  fr'$N - M < a \leq N + M$' + '\n\n' 
##                + fr'$H = {H}$' + '\n\n' 
##                + r'$N = [e^{21}]$' + '\n\n' 
##                + fr'$M = 10^5$' + '\n\n' 
##                + fr'$\lambda = H/(\log N - 1) = {H*p:.5f}$' + '\n\n' 
##                + r'$\mathbb{E}[X] = $' + f'{mu:.5f}' + '\n\n' 
##                + r'$\mathrm{Var}(X) = $' + f'{sigma:.5f}' + '\n\n' 
##                + fr'median : ${med}$' + '\n\n' 
##                + fr'mode(s): ${modes}$', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5), transform=ax.transAxes)
##    # Formating/labeling
##    ax.set_xticks(hor_axis)
##    ax.set_xlabel(r'$m$ (number of primes in an interval)')
##    ax.set_ylabel('prop\'n of intervals with' + r' $m$ ' + 'primes')
##    ax.legend(loc=2, ncol=1, framealpha=0.5)
##
##    # A grid is helpful, but we want it underneath everything else. 
##    ax.grid(True,zorder=0,alpha=0.7)   
##    
### Generate the animation
##X_anim = animation.FuncAnimation(fig, plot, frames=C, interval=100, blit=False, repeat=False)
##
### This is supposed to remedy the blurry axis ticks/labels. 
##plt.rcParams['savefig.facecolor'] = 'white'
##
##plot(C[-1])
##plt.show()    

# EXTENSIONS

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


