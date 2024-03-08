# SIMD-Single-Instruction-Multiple-Data-

Parallel Computing and Data Processing

Moore’s law was the idea that computers double in efficiency roughly every two years, leading to exponentially more computing power over time. This was true for a very long time. However, sometime in the last decade, computer cores have stopped getting faster. The answer to the “end of Moore’s Law” is Parallel Computing. The goal of this project is to gain experience working with different ways to parallelize and distribute computation. Specifically, we will look at two techniques: parallelism within a single core (i.e., SIMD) and parallelism with multiple independent processes (i.e., distributed computing).

This repository contains a simple SIMD program to compute the prediction for an entire test set for a linear regression model. Concretely, I compared two implementations:

- A standard single-thread scalar version of the program.
- A single-thread SIMD version of the function. For this function, I use the CPU intrinsics to execute the SIMD code. That is, I cannot use compiler optimizations to perform automatic vectorization.


The project was solely done with C++ 17 using CMake to compile and build the program. 
All commands were executed in terminal in a Mac OS 2019, intel x86-64 processor. 

Once succesfully executed locally, I remotely conneced via ssh to a linux machine of KU Leuven Computer Science department. 
I compiled and executed the program thre sucesfully as well. For modifications of the files I used 'nano'.

The code is not my property. Part of the code was provided by the BDAP team of the engineering department of KU Leuven. Warm regards, Nikolaos Kales
