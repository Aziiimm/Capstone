/**
 * Kernel 2: Matrix Factorization via Stochastic Gradient Descent (SGD)
 *
 * Target: Training the core recommender system using Collaborative Filtering.
 *
 * Description:
 * This kernel updates user and item latent factor embeddings simultaneously
 * based on actual review ratings (and sentiment).
 *
 * Difficulty: 85 / 100
 * - Pros: Pure linear algebra and vector math, which GPUs excel at.
 * - Cons: Race conditions! Multiple threads might try to update the same item
 * embedding simultaneously if multiple users reviewed the same item in the same
 * batch. Requires careful use of atomicAdd or lock-free parallel execution
 * strategies (like Graph Coloring).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Updates user and item embeddings using SGD for Matrix Factorization.
 *
 * @param d_users        Array of user IDs for the current batch
 * @param d_items        Array of item IDs for the current batch
 * @param d_ratings      Array of actual ratings for the user-item pairs
 * @param d_user_embeds  Flattened array of user embeddings (size: num_users *
 * embed_dim)
 * @param d_item_embeds  Flattened array of item embeddings (size: num_items *
 * embed_dim)
 * @param num_pairs      Number of interactions in this batch
 * @param embed_dim      Dimensionality of the latent factors
 * @param learning_rate  Alpha parameter for gradient descent
 */
__global__ void
sgdMatrixFactorizationUpdate(const int *d_users, const int *d_items,
                             const float *d_ratings, float *d_user_embeds,
                             float *d_item_embeds, int num_pairs, int embed_dim,
                             float learning_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_pairs) {
    int u = d_users[idx];
    int i = d_items[idx];
    float actual_rating = d_ratings[idx];

    // TODO: Implement SGD Update
    // 1. Compute predicted rating: dot product of d_user_embeds[u] and
    // d_item_embeds[i]
    // 2. Compute error: actual_rating - predicted_rating
    // 3. Compute gradients for user and item embeddings
    // 4. Apply updates securely using atomicAdd to avoid race conditions:
    //    atomicAdd(&d_item_embeds[i * embed_dim + f], gradient_i);
  }
}
