---
slug: inverting_matrices_in_place_addenda
title: Inverting Matrices In-Place Addenda
authors: [Silicon42]
tags: [Math, Linear-Algebra, Programming, Optimization, CPU-Architecture]
---

It occurred to me that I didn't really explain very well how to check the 
dependencies of the individual parts of the computation in my previous post so 
in addition to the code, I went and made a diagram or two to illustrate it a bit 
better. <!-- truncate --> This is a continuation of the previous post, for full 
context, read that first. The scope of this project grew just a bit since last 
time, so it's now a full fledged repo 
[available here](https://github.com/Silicon42/Cholesky_decomp_matrix_inversion). 
I also wrote a hand unrolled version for use comparing how different compilers 
handled it with [godbolt.org]. More on that later.

## Cholesky Decomposition Dependency Graph

Below is a diagram of the element-wise dependencies for the Cholesky decomposition,
with the "S" column being the intermediate calculations and the "L or D" column 
being the final decomposed values, they still represent the SAME slots. The 
diagram progresses mostly right to left, bottom to top because that's just how I 
drew it and I'm not drawing it again just to fix that. Draw.io is nice but it's 
not the most convenient to use in my opinion, especially when their splines just 
seem to randomly get additional nodes at their ends that you can't get rid of 
when you try to move them sometimes.

![cholesky decomposition dependency graph](https://raw.githubusercontent.com/Silicon42/Cholesky_decomp_matrix_inversion/refs/heads/main/cholesky%20decomp%20dependency.svg)

An arrow pointing into a box means that box depends on the box the arrow came from. 
Same colored arrows are required at the same time, or in other words they take part 
in the same calculation. All values in S have an implicit dependency on the "A" 
coefficients at their own position as that's what they are initialized to. Dashed 
arrows represent a bundle of arrows of multiple colors with the last required being 
the primary color. A box is ready to fulfill any outgoing dependencies that rely 
on it onces all of it's incoming arrows are accounted for. Boxes with multiple 
arrow pairs keep intermediate sum results in them and are not usable for later 
dependencies until all incoming arrows have been accounted for. Once a box in the 
"L or D" column has been written to, the corresponding box in the "S" column can 
no longer be used because it has been overwritten since it is the same slot, hence 
the need for a temporary variable.

## Access Pattern

I'll come back and add an animation of the access pattern in a later edit.

For the purpose of 5x5 matrices, the exact access pattern isn't as important 
since almost everything should be in registers or prefetched early enough to not
matter, assuming the compiler is smart enough, however if you look at the above 
dependency graph, there are several items being changed in a single step, these 
are because the order doesn't matter as their results don't affect each other. 
Theoretically this means that hardware with vector processing capability could 
be used to actually do those steps simultaneously if the linear ordering were 
reworked so that they were adjacent and conformed to alignment requirements, and 
for n smaller than the hardware vector length, could shave the O(n^3) running 
time of the algorithm to practically O(n^2). You would have to add additional 
masking to only set results that actually correspond to real calculation items 
but it could improve floating point calculation throughput. Unfortunately, there 
is also likely very little benefit for small n since out-of-order-execution and 
instruction-level-parallelism would probably also pick up on this and give an 
equivalent benefit without as much effort.

Speaking of out-of-order-execution, your compiler and possibly the CPU as well 
may rearrange some of the non-interdependent operations and therefore accesses. 
This is done to prevent pipeline stalls since loading from memory and floating 
point operations can take many cycles to complete, therefore they should ideally
be arranged to not need the result for another calculation immediately after. 
Unfortunately, in floating point land, (A + B) + C is not quite the same as A + 
(B + C) so a compiler will not typically re-order one to the other and a CPU 
definitely won't even if, say, A is still being calculated but it has an adder 
waiting to be used. This means that we want to try to keep that in mind when 
considering iteration order, such as when taking the reciprocal of one of the 
diagonal elements, it is better to not immediately use the result for calculating 
the current columns L values and decrementing the column, and instead use the 
previously calculated diagonals first while it's computing and increment instead.
I haven't yet fully explored which ordering is optimal for this yet and I'm not 
sure I'll ever get the time to come back to it, but it is something interesting 
to keep in mind.

On a related note, summing the L and S products separately, using more temporary 
variables could possibly be faster and more accurate than the current 
implementation if it can manage to avoid some of the dependencies that trying to 
force the computation into as little memory as possible introduced, such as 
multiple columns waiting on the temporary variable to be free so they can take 
their turn using it. It could improve the accuracy if the additions happened 
such that similar magnitude values got added first or at least tended to be.

# Unrolling and Compilers

I mentioned before that I tried out Godbolt for the first time, to get some 
sense of if my optimizations were actually effective and from what little I can 
tell without running an actual benchmark, they do seem to be since on most 
compilers all of the calculations stay in registers without going back and forth 
attempting to load or store to memory. I also learned that for loops over very 
small amounts of instructions, unrolling can actually make the program shorter 
in addition to faster if it's the case that the index calculating overhead is 
more than the actual calculation, it also helps that since all the array indexes
become fixed values, the compiler doesn't have to worry about loading and storing
to memory locations or register shuffling so that each iteration uses the same 
register. In other words, this piece of code really must be unrolled to be truly 
performant.

Comparing the results of a few different compilers was interesting as well. 
Clang had no problems unrolling the loop with just a command line option and 
regularly created nice relatively short assembly. GCC typically performed on par 
with Clang however it needed additional pragmas to encourage it to actually 
unroll the loops despite the unrolling making the code both shorter and faster. 
From my brief look into it this might be because it only tries to unroll 
innermost loops but that's just based on what I saw someone mention so I'm not 
sure if that's the whole story or not. MSVC on the other hand didn't seem to
want to unroll it at all, resulting in assembly that was anywere from around
2 to 5 times as long as GCC or Clang on the same architecture. Maybe there is a 
way to get it do so, but I couldn't figure it out. Admittedly I didn't do much 
looking into the issue since I've been mostly using GCC recently anyway. I also 
tried AVR GCC and realized that having access to hardware floating point is not 
something I should ever take for granted as the assembly length ballooned more 
than an order of magnitude.

# Source Code

Here's the easily viewable code I promised in the last post, the full comments 
have been somewhat shortened from what's in the Git and any changes to the Git 
won't be reflected here from here on out so if you want the latest version, get 
it [here](https://github.com/Silicon42/Cholesky_decomp_matrix_inversion). For
those who just want a quick reference of a working version it's available below.

```c
// MIT License: Copyright (c) 2025 Curtis Stofer a.k.a. Silicon42

#define TRI_INDEX(i,j)    (((int)(i)*((int)(i) + 1))/2 + (int)(j))

//easy type changing if you want another width of float
typedef float generic_float;

/*
Expects a representation of a symmetric positive definite matrix lower half 
packed in a linear array such that the index is as follows.
 0
 1  2
 3  4  5
 6  7  8  9
10 11 12 13 14

Does not check to verify the matrix satisfies these conditions.

Uses the L*D*L^T version of the Cholesky decomposition where D is a diagonal 
matrix and L is a lower unit triangular matrix.

Returns the coefficients of L**-1 in the lower triangle and the coefficiensts of
D**-1 on the diagonal
*/
void cholesky_inv_sym_5(generic_float A[15])
{
    // GCC only unrolls deepest loops but for this, unrolling is both faster
    // and results in a shorter program
    #pragma GCC unroll 9
    // for each row
    for(int i = 1; i < 5; ++i)
    {
        // calculate the D**-1 coeff, not stored as D coeff
        // as a speed/accuracy tradeoff, 
        A[TRI_INDEX(i,-1)] = 1 / A[TRI_INDEX(i,-1)];
        // for each element in the current row before the diagonal
        for(int j = 0; j < i; ++j)
        {   // calculate the -L coefficient, negated b/c it's only ever
            // needed as a negative stored in temp variable so as to not
            // overwrite the cell before using its S element
            generic_float L_temp = -A[TRI_INDEX(i,j)] * A[TRI_INDEX(j,j)];
            for(int k = i; k < 5; ++k)
                A[TRI_INDEX(k,i)] += L_temp * A[TRI_INDEX(k,j)];

            A[TRI_INDEX(i,j)] = L_temp;
        }
    }
    // get reciprocal of the last D coeff to convert to D**-1 coeff
    A[TRI_INDEX(4,4)] = 1 / A[TRI_INDEX(4,4)];

    // now that the L coefficients are fully calculated, they can be inverted
    #pragma GCC unroll 9
    // for each row, bottom first
    for(int i = 3; i > 0; --i)
        // for each element below the diagonal, right-most first
        for(int j = i - 1; j >= 0; --j)
            for(int k = i + 1; k < 5; ++k)    // for each element below [i][j]
                A[TRI_INDEX(k,j)] += A[TRI_INDEX(i,j)] * A[TRI_INDEX(k,i)];
}
```