---
slug: inverting_matrices_in_place
title: Inverting Matrices In-Place
authors: [Silicon42]
tags: [Math, Linear-Algebra, Programming, Optimization, CPU-Architecture]
---

Over the last couple days I've been working out how to solve overdetermined 
systems of relatively short linear equations as part of a program that needs to 
fit conics, specifically ellipses. <!-- truncate -->

Code for the following post can be found in 
[this Gist](https://gist.github.com/Silicon42/45a59f04e767561b1c8bcfabe6cab48c).
As a side note, I also learned that GitHub doesn't track Gist commits in their
commit graph because that's where I've been putting the code for this 
particular sub-project. This is despite not mentioning it in their "why might 
your commits not be showing up" page and despite clearly having that data so 
when you only commit to gists you end up with a blank hole in your graph. 
:frowning: 

Yes, I'm aware there are already libraries for that kind of thing in every 
language concievable and they are typically more efficient than anything a 
single dev can come up with on their own due to how widely used they tend to 
be. No, I did not use one of them.

The reason I decided to write my own is because those libraries tend to assume 
that you are working with very big linear equations with many unknowns, which 
results in very big matrices that are typically dynamically allocated and by 
fact of their sheer size are usually not capable of fitting fully in L1 cache 
and thus would not significantly benefit from optimizations on that front. I 
only need it to work on 5 unknowns and because I also need it to be capable of 
working fully within OpenCL kernel code without host intervention, I can't use 
dynamic memory allocation. For this particular problem, such libraries are over 
generalized and a specialized approach is warranted since this is a decently 
large chunk of computation on the critical codepath.

Since I will be working on systems of 5 unknowns, solving them will require 
inverting a 5x5 matrix, and since it is an overdetermined system, this comes as 
part of taking the pseudo-inverse. This process is a form linear regression and 
isn't directly relevant to the topic at hand but it does guarantee that the 
resulting 5x5 will be a symmetric, positive semi-definite matrix that we'll call 
$\bm{A}$. I'll probably talk more about that in a later post, but if not there 
are plenty of other decent explanations out there.

## Some Math Background
For a fully in depth description of what's involved, we can rely on the 
[Wikipedia article on Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) 
however that's not entirely focussed on just what we'll need for this. 
Paraphrasing a bit, the most relevant parts are as follows.

That $\bm{A}$ is "symmetric" means symmetric across the diagonal, such that 
$\bm{A_{12}} = \bm{A_{21}}$, so in this case, 10 of the entries would be 
duplicates and can be ommitted. The "positive semi-definite" part means 
something about all the eigen values being non-negative but for the current 
goal, we only need to know that it makes it qualify for Cholesky decomposition 
which is more efficient than LU decomposition. I choose to live dangerously and 
not check that any of the eigenvectors is $0$ because that is extremely rare in 
real data for overdetermined systems and it's not critical to my use case if 
there is the extremely rare edge case that the computation fails because of 
this. 

So in more formulaic terms, we want to solve 
$$
\bm{A\vec{x}}=\bm{\vec{b}}
$$
where $\bm{\vec{x}}$ is the column vector representing the unknowns and 
$\bm{\vec{b}}$ is the column vector representing the constant factors, so we 
multiply both sides by $\bm{A}^{-1}$ to get
$$
\bm{\vec{x}}=\bm{A}^{-1}\bm{\vec{b}}
$$
Now Cholesky decomposition states that if we have a symmetric positive definite 
matrix, we can break it down into the form
$$
\bm{A}=\bm{LDL}^T
$$
where $\bm{D}$ is a diagonal matrix with only values on its diagonal with all 
else being $0$s, and $\bm{L}$ is a lower unit triangular matrix, meaning the 
lower-left triangle has values, the upper-right is filled with $0$s, and the 
diagonal is filled with $1$s.

We choose this decomposition because it's element calculation doesn't require 
square roots and because diagonal and unit triangular matrices are much easier 
to invert than most other more general matrices. The inverse of a diagonal 
matrix is simply a diagonal matrix with the reciprocal of the corresponding 
elements and unit triangular matrices are in the perfect form for using 
Gauss-Jordan elimination to invert and the result will also be a unit triangular 
matrix of the same shape, giving
$$
\bm{A^{-1}}=(\bm{L}^{-1})^T\bm{D}^{-1}\bm{L}^{-1}
$$
The individual entries of $\bm{D}$ and $\bm{L}$ can be calculated according to 
the following recurrence relation
$$
\bm{D_i}=\bm{A_{ii}}-\sum_{k=1}^{j-1}\bm{L_{ik}D_k L_{ki}}
$$
$$
\bm{L_{ij}}=\frac{\bm{A_{ij}}-\sum\limits_{k=1}^{j-1}\bm{L_{ik}D_k L_{kj}}}{\bm{D_i}}
$$

## Math Simplifications
Here's where it starts deviating from the Wikipedia entry.

Those last two equations look very similar and seem to have a redundant 
multiplication in them whose only job is to cancel a previous division. That 
might be there for the case that someone is using them to recursively break down 
a much larger matrix, but we don't care about that for this, we can instead 
rewrite it using an intermediate matrix $\bm{S}$ to streamline the computations 
to not have completely separate logic for nearly identical formulas
$$
\bm{S_{ij}} = \bm{A_{ij}}-\sum_{k=1}^{j-1}\bm{S_{ik}L_{kj}}
$$
$$
\bm{D_i} = \bm{S_{ii}}
$$
$$
\bm{L_{ij}} = \bm{S_{ij}}/\bm{D_i}
$$
Removing the muliplication of dividends and then re-multiplication by the 
divisor should prevent a small amount of floating point inaccuracy from creeping 
in as well as reducing the number of multiplies.

## Decomposition In-Place
You might think that introducing an intermediate matrix would significantly 
increase the amount of memory needed for this but this isn't the case. Elements 
of $\bm{D}$ map exactly onto the diagonal of $\bm{S}$ and elements of $\bm{L}$ 
are only dependent on their matching column diagonal and their corresponding 
element of $\bm{S}$. Since the diagonal elements of $\bm{L}$ are known to be 
$1$s, we can let them be implicit and instead share the space when overwriting 
$\bm{S}$. The only computations dependent on elements of $\bm{A}$ are the 
corresponding elements of $\bm{S}$, we can safely overwrite them with the 
results of $\bm{S}$, with only a slight caveat that we need a single temporary 
variable when calculating the diagonals since they need both $\bm{S_{ik}}$ and 
$\bm{L_{ik}}$ at the same time yet $\bm{L_{ik}}$ wants the result of its 
calculation to overwrite $\bm{S_{ik}}$.

## Hot Cache
With all of this fitting into the same space as the original matrix, we only 
need 15 floats +1 temporary float, which means that if we are using 32-bit 
floats, the whole thing fits in 64 bytes, or a single cache line (although it's 
more likely split across 2 if it's just on the stack). This can be arranged by 
not storing the other triangle at all in a linear array of 15 elements that you 
index as follows
$$
n = i*(i+1)/2 + j
$$
and having the compiler unroll it to avoid the integer operations indexing that 
way would cost. Alternatively, one could use a set of individual variables on 
the stack and manually unroll it. Since all access touch the same 16 locations, 
we can easily pre-fetch the contents from memory to ensure at minimum L1 cache 
lookup times or in the case that the 15 values were just computed on the stack 
we automatically get that for free. Depending on how many hardware floating 
point registers one has, the whole thing might not need to even access L1. This 
means no inefficiencies waiting around for cache misses and virtually zero 
chance of causing or being susceptable to eviction related stalls.

## Inversion In-Place
Now that we need to calculate the inverses we can see that the elements of 
$\bm{D}$ are only ever used to divide, this means we can choose to either 
prioritize accuracy or speed. To prioritize accuracy, don't invert $\bm{D}$ and 
opt to simply apply the element-wise division to the column vector result of 
$\bm{L}^{-1}\bm{\vec{b}}$. To prioritize speed, invert $\bm{D}$ early when doing 
the decomposition and multiply by the reciprocal instead of dividing when 
calculating the elements of $\bm{L}$. This leads to a small loss in accuracy 
which compounds with increasing row index but in such a small matrix, this is 
perfectly acceptable. The reason this is faster is because floating point divide 
is typically 2x as slow or worse than floating point multiply. Depending on CPU 
architecture and compiler support, taking a reciprocal may also be faster than a 
divide, so we trade 15 divides, 5 from the application of the inverse to the 
column vector + 10 from calculating the elements of $\bm{L}$, for 5 reciprocals 
and 15 multiplies.

When we go to invert $\bm{L}$ using 
[Gauss-Jordan elimiation](https://en.wikipedia.org/wiki/Gaussian_elimination#Gauss%E2%80%93Jordan_elimination), 
we subtract multiples of one row (or column since 
$(\bm{L}^{-1})^T=(\bm{L}^T)^{-1}$) from another. I'll use a 3x3 Matrix augmented 
with the identity matrix for ease of illustration.
$$
\begin{array}{rl}
\begin{array}{}
R_1\\
R_2\\
R_3
\end{array} & 
\left[\begin{array}{rrr|rrr}
1 & 0 & 0 & 1 & 0 & 0\\
a & 1 & 0 & 0 & 1 & 0\\
b & c & 1 & 0 & 0 & 1
\end{array}\right] \\ \\
\begin{array}{}
\\
R_2-=aR_1\\
R_3-=bR_1
\end{array} & 
\left[\begin{array}{rrr|rrr}
1 & 0 & 0 &  1 & 0 & 0\\
0 & 1 & 0 & -a & 1 & 0\\
0 & c & 1 & -b & 0 & 1
\end{array}\right] \\ \\
\begin{array}{}
\\
\\
R_3-=cR_2
\end{array} &
\left[\begin{array}{rrr|rrr}
1 & 0 & 0 &    1 &  0 & 0\\
0 & 1 & 0 &   -a &  1 & 0\\
0 & 0 & 1 & ca-b & -c & 1
\end{array}\right]
\end{array}
$$
As you can see, doing this vacates the elements on the left half at the same 
time it populates the right with the negative of it, meaning that this step can 
also be done in place. We can skip this first negation step entirely if when 
computing the elements of $\bm{L}$ we take into account that we only ever need 
the negative of the elements and so just store $-\bm{L}$ instead. Further 
summations after initial population are the result of the product of the last 
fully computed row above with the element in the current row with matching 
column index.

## Conclusion and Code
With all that, the computation is done. There might be some ways to eke out just 
a bit more performance with vector processing but I'm not sure the increased 
space requirements would warrant that on such a small matrix size. If people ask 
for benchmarks of this I'd be willing to try and set some up but for now I'm 
good with this being based solely on theory. Once again the code is available in 
[this Gist](https://gist.github.com/Silicon42/45a59f04e767561b1c8bcfabe6cab48c).

I'll edit this article soon to provide a more accessible an at-a-glance version 
without such large comment blocks.
