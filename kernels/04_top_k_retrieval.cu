/**
 * Kernel 4: Top-K Retrieval and Ranking Metrics
 *
 * Target: Extracting the top K recommendations per user and evaluating NDCG/Hit
 * Ratio.
 *
 * Description:
 * Sorting the entire prediction matrix is extremely slow. This kernel operates
 * a thread block per user to maintain a local register-based or shared-memory
 * min-heap to extract only the top-K items efficiently.
 *
 * Difficulty: 80 / 100
 * - Pros: Massively reduces data transfer times from GPU to CPU for evaluation.
 * - Cons: Implementing an efficient parallel min-heap or bitonic sort using
 * thread block synchronization (`__syncthreads()`) is notoriously tricky to get
 * right for performance.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Retrieves the top K items for a specific user and calculates Hit
 * Ratio.
 *
 * @param d_predictions     Dense 2D array of predictions (num_users x
 * num_items)
 * @param d_ground_truth    Array of actual held-out items the user interacted
 * with
 * @param d_top_k_items     Output array storing top K item IDs per user
 * @param K                 Number of items to retrieve (e.g., 10 or 50)
 * @param num_items         Total number of items in the catalog
 */
__global__ void retrieveTopKAndEvaluate(const float *d_predictions,
                                        const int *d_ground_truth,
                                        int *d_top_k_items, int K,
                                        int num_items) {
  // One block per user
  int user_id = blockIdx.x;
  int local_tid = threadIdx.x;

  // Pointer to predictions for this specific user
  const float *user_preds = &d_predictions[user_id * num_items];

  // TODO: Implement block-wide Top-K Search
  // 1. Each thread loads and processes a subset of items, keeping a local Top-K
  // array.
  // 2. Use block shared memory (`__shared__`) to reduce the local Top-K arrays
  // into an overall Block Top-K.
  // 3. Thread 0 writes the final top K items to `d_top_k_items`.
  // 4. Thread 0 optionally checks if `d_ground_truth[user_id]` is in the top K
  // to update HR@K metrics.
}
