/*
 * Copyright 2022 BDAP team.
 *
 * Author: Laurens Devos
 * Version: 0.1
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <immintrin.h>
#include <xmmintrin.h> // For _mm_malloc and _mm_free
#include <memory>
#include <cstdlib>


//using std::chrono::steady_clock;
using std::chrono::high_resolution_clock; // better precision
using std::chrono::microseconds;
using std::chrono::duration_cast;

/**
 * A matrix representation.
 *
 * Based on:
 * https://github.com/laudv/veritas/blob/main/src/cpp/basics.hpp#L39
 */


template<typename T> // checks for alignmetn
bool is_aligned(T* ptr, std::size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
class aligned_allocator : public std::allocator<T> {
public:
    using size_type = typename std::allocator<T>::size_type;
    using pointer = typename std::allocator<T>::pointer;
    using const_pointer = typename std::allocator<T>::const_pointer;

    // Default constructor
    aligned_allocator() noexcept : std::allocator<T>() {}

    template <class U>
    aligned_allocator(const aligned_allocator<U>&) noexcept {}

    pointer allocate(size_type num, const void* hint = 0) {
        if (auto p = static_cast<pointer>(_mm_malloc(num * sizeof(T), 32))) {
            return p;
        }
        throw std::bad_alloc(); // error handling: throw an exception with more context
    }

    void deallocate(pointer p, size_type num) {
        _mm_free(p);
    }
};


template <typename T>
struct matrix { // structure matrix
private:
    //std::vector<T> vec_;
    // aligned_allocator with std::vector
    std::vector<T, aligned_allocator<T>> vec_;


public:
    size_t nrows, ncols;
    size_t stride_row, stride_col; // in num of elems, not bytes

    /** Compute the index of an element. */
    inline size_t index(size_t row, size_t col) const
    {
        if (row >= nrows)
            throw std::out_of_range("out of bounds row");
        if (col >= ncols)
            throw std::out_of_range("out of bounds column");
        return row * stride_row + col * stride_col;
    }

    /** Get a pointer to the data */
    inline const T *ptr() const { return vec_.data(); }

    /** Get a pointer to an element */
    inline const T *ptr(size_t row, size_t col) const
    { return &ptr()[index(row, col)]; }

    /** Get a pointer to the data */
    inline T *ptr_mut() { return vec_.data(); }

    /** Get a pointer to an element */
    inline T *ptr_mut(size_t row, size_t col)
    { return &ptr_mut()[index(row, col)]; }

    /** Access element in data matrix without bounds checking. */
    inline T get_elem(size_t row, size_t col) const // get
    { return ptr()[index(row, col)]; }

    /** Access element in data matrix without bounds checking. */
    inline void set_elem(size_t row, size_t col, T&& value) // set
    { ptr_mut()[index(row, col)] = std::move(value); }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](size_t i) const
    { return ptr()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T& operator[](size_t i)
    { return ptr_mut()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](std::pair<size_t, size_t> p) const
    { auto &&[i, j] = p; return get_elem(i, j); }

    /*matrix(std::vector<T>&& vec, size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(std::move(vec))
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}*/

    matrix(std::vector<T, aligned_allocator<T>>&& vec, size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(std::move(vec))
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}

    matrix(size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(nr * nc)
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)// detertmines how many elements are in memory
        , stride_col(sc) {}
};

using fmatrix = matrix<float>;

std::tuple<fmatrix, fmatrix, fmatrix, float>
read_bin_data(const char *fname) // reads the .bin files
{
    std::ifstream f(fname, std::ios::binary);

    char buf[8];
    f.read(buf, 8);

    int num_ex = *reinterpret_cast<int *>(&buf[0]);
    int num_feat = *reinterpret_cast<int *>(&buf[4]);

    std::cout << "num_ex " << num_ex << ", num_feat " << num_feat << std::endl;

    size_t num_numbers = num_ex * num_feat;
    fmatrix x(num_ex, num_feat, num_feat, 1); // for features
    fmatrix y(num_ex, 1, 1, 1); // for labels
    fmatrix coef(num_feat, 1, 1, 1); // for coefficients

    f.read(reinterpret_cast<char *>(x.ptr_mut()), num_numbers * sizeof(float)); // read content from .bin file
    f.read(reinterpret_cast<char *>(y.ptr_mut()), num_ex * sizeof(float));
    f.read(reinterpret_cast<char *>(coef.ptr_mut()), num_feat * sizeof(float));

    f.read(buf, sizeof(float));
    float intercept = *reinterpret_cast<float *>(&buf[0]); // float point (intercept) of linear regr. model

    return std::make_tuple(x, y, coef, intercept); // return x,y,coef matrices and intercept in a form of a tuple
}

fmatrix evaluate_scalar(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);

    // TODO implement this method using regular C++
    //
    // You CANNOT use threads.
    // We are forbidding multithreading to make the coding take less time and
    // get you to focus on learning the SIMD part.
    for (size_t i = 0; i < x.nrows; ++i) {
        float sum = 0.0;
        for (size_t j = 0; j < x.ncols; ++j) {
            sum += x.get_elem(i, j) * coef[j];
        }
        output[i] = sum + intercept;
    }

    return output;
}

fmatrix evaluate_simd(fmatrix x, fmatrix y, fmatrix coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);
    const size_t alignment = 32;

    // TODO implement this method using SIMD intrinsic functions. See the second
    // exercise session.
    //
    // You CANNOT use threads.
    // We are forbidding multithreading to make the coding take less time and
    // get you to focus on learning the SIMD part.

    if (!is_aligned(x.ptr(), alignment)) {
        std::cerr << "Matrix data is not aligned. Adjusting for optimal SIMD performance.\n";
        // Consider options to handle this, like using unaligned loads or aligning the data
    }

    for (size_t i = 0; i < x.nrows; ++i) {
        __m256 sum_vec = _mm256_setzero_ps(); // Initialize sum vector to 0
        int j = 0, ncols8 = (x.ncols / 8) * 8;

        // Report: Loop is not vectorized: loop control flow is not understood by vectorizer [-Rpass-analysis=loop-vectorize]
        for (; j < ncols8; j += 8) {  
            __m256 x_vec = _mm256_loadu_ps(x.ptr(i, j)); // Load 8 elements from x   --> (I use 'u' for unasigned as seen in exercise 3)
            __m256 coef_vec = _mm256_loadu_ps(coef.ptr(0, j)); // Load 8 coefficients
            __m256 prod_vec = _mm256_mul_ps(x_vec, coef_vec); // Multiply x and coef
            sum_vec = _mm256_add_ps(sum_vec, prod_vec); // Accumulate products

            /*__m256 x_vec = _mm256_load_ps(x.ptr(i, j)); // Load 8 elements from x   --> (I delete 'u' for unasigned, to do memory aligned)
            __m256 coef_vec = _mm256_load_ps(coef.ptr(0, j)); // Load 8 coefficients
            __m256 sum_vec = _mm256_fmadd_ps(x_vec, coef_vec, sum_vec); // Fused multiply-add*/
        }

        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec); // Store sum_vec in an sum_array
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                    sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7]; // get total sum

        // Report: Loop is not vectorized: loop control flow is not understood by vectorizer [-Rpass-analysis=loop-vectorize]
        // Handle remaining elements
        for (; j < x.ncols; ++j) { 
            sum += x.get_elem(i, j) * coef[j];
        }

        output[i] = sum + intercept; // add the intercept
    }        
    return output;
}

int main(int argc, char *argv[])
{
    // These are four linear regression models
    auto &&[x, y, coef, intercept] = read_bin_data("/Users/nikolaos/KU_Leuven/Summer/BDA/Assignment_2/data/calhouse.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/allstate.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/diamonds.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/cpusmall.bin");

    // This is a logistic regression model, but can be evaluated in the same way
    // All you would need to do is apply the sigmoid to the values in `output_*`
    //auto &&[x, y, coef, intercept] = read_bin_data("data/mnist_5vall.bin");
    
    // TODO repeat the number of time measurements to get a more accurate
    // estimate of the runtime.

    // Measuring the times a few time for more accurate runtime
    const int iterations = 10000;; // Number of iterations for the measurements
    const int warmup_iterations = 100;

    // Warm-up iterations to warm up the system and ensure caches are populated
    // The warm-up iterations using evaluate_simd ensure that both functions will have similar starting conditions regarding CPU cache.
    for (int i = 0; i < warmup_iterations; ++i) {
        evaluate_simd(x, y, coef, intercept);
    }

    double total_scalar_time = 0.0, total_simd_time = 0.0;

    /* I use Separate loops for consistency. According to literature separate loops are prefered when comparing overall performance of two separate implementations*/
    //Scalar
    for (int i = 0; i < iterations; ++i) {
        auto tbegin_scalar = high_resolution_clock::now();
        auto output_scalar = evaluate_scalar(x, y, coef, intercept);
        auto tend_scalar = high_resolution_clock::now();
        total_scalar_time += duration_cast<microseconds>(tend_scalar - tbegin_scalar).count();
    }

    //SIMD
    for (int i = 0; i < iterations; ++i) {
        auto tbegin_simd = high_resolution_clock::now();
        auto output_scalar = evaluate_simd(x, y, coef, intercept);
        auto tend_simd = high_resolution_clock::now();
        total_simd_time += duration_cast<microseconds>(tend_simd - tbegin_simd).count();
    }

    // Compute averages
    double avg_scalar_time = total_scalar_time / iterations / 1000.0; // Convert to milliseconds
    double avg_simd_time = total_simd_time / iterations / 1000.0;

    std::cout << "Average evaluated scalar time: " << avg_scalar_time << "ms\n";
    std::cout << "Average evaluated SIMD time: " << avg_simd_time << "ms\n";
}


