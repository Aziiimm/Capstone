/**
 * Kernel 1: Sentiment Analysis & Tokenization
 * 
 * Target: GPU-accelerated NLP processing of raw review text.
 * 
 * Description:
 * This kernel will take raw character arrays (extracted from strings) and perform parallel tokenization
 * or direct sentiment scoring (e.g., VADER rules or custom dictionary lookup) without passing the strings
 * individually back to the CPU.
 * 
 * Difficulty: 65 / 100
 * - Pros: Highly independent work per thread (embarrassingly parallel).
 * - Cons: Variable length strings can cause thread divergence and unbalanced workloads. String
 *   manipulation in CUDA C++ requires careful memory management.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Computes a sentiment score per review.
 *
 * @param d_text_data       Pointer to flatten character array containing all reviews
 * @param d_offsets         Pointer to array of offsets indicating where each review starts
 * @param d_sentiment_out   Output array storing the float sentiment score for each review
 * @param num_reviews       Total number of reviews to process
 */
__global__ void computeSentimentScores(const char* d_text_data, const int* d_offsets, float* d_sentiment_out, int num_reviews) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_reviews) {
        // Find bounds for the current string
        int start_idx = d_offsets[idx];
        int end_idx = d_offsets[idx + 1];
        
        // TODO: Implement tokenization and dictionary lookup
        // 1. Iterate over characters from start_idx to end_idx
        // 2. Identify words (split by spaces/punctuation)
        // 3. Look up words in a device-loaded sentiment dictionary
        // 4. Calculate total score and write to d_sentiment_out[idx]
        
        d_sentiment_out[idx] = 0.0f; // Placeholder
    }
}
