/**
 * Kernel 5: Graph Aggregation / Message Passing
 *
 * Target: Advanced representations using Graph Neural Networks (e.g., LightGCN,
 * GraphSAGE).
 *
 * Description:
 * Aggregating embeddings from neighbors in a user-item bipartite graph. This
 * kernel acts as a specialized Scatter/Gather operation taking into account
 * edge weights (sentiment scores).
 *
 * Difficulty: 90 / 100
 * - Pros: Enables state-of-the-art recommendation utilizing complex multi-hop
 * graph paths.
 * - Cons: The most difficult memory access pattern. "Power-law" distributions
 * mean some users or items have millions of connections while others have 1.
 * This causes massive load imbalance and requires sophisticated warp-scheduling
 * or segmented scans to prevent thread blocks from stalling indefinitely.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Aggregates neighbor embeddings for nodes in a Graph Neural Network
 * architecture.
 *
 * @param d_node_embeds     Input embeddings for all nodes (users + items)
 * @param d_edge_src        Source node of each edge
 * @param d_edge_dst        Target node of each edge
 * @param d_edge_weights    Weight of each edge (e.g., sentiment score)
 * @param d_out_embeds      Output aggregated embeddings
 * @param num_edges         Total number of edges in the graph
 * @param embed_dim         Dimensionality of the embeddings
 */
__global__ void
gnnMessagePassingAggregate(const float *d_node_embeds, const int *d_edge_src,
                           const int *d_edge_dst, const float *d_edge_weights,
                           float *d_out_embeds, int num_edges, int embed_dim) {
  // Typically parallelized over edges or destination nodes
  int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (edge_idx < num_edges) {
    int src = d_edge_src[edge_idx];
    int dst = d_edge_dst[edge_idx];
    float weight = d_edge_weights[edge_idx];

    // TODO: Implement scatter/gather aggregation
    // 1. Load the embedding of the source node `src`.
    // 2. Multiply it by the edge `weight` (sentiment strength).
    // 3. Safely add the weighted embedding to the destination node `dst` in
    // `d_out_embeds`.
    //    (Requires atomicAdd for multidimensional floating-point arrays).

    // Example:
    // for(int f = 0; f < embed_dim; ++f) {
    //     atomicAdd(&d_out_embeds[dst * embed_dim + f], d_node_embeds[src *
    //     embed_dim + f] * weight);
    // }
  }
}
