### **2. Updated WEEK2.md**

# Weekly Check-in Notes (Week 2)

- **Discussed the implementation of a Hybrid Recommendation System:** We are moving beyond simple ratings to include Sentiment Analysis. We confirmed that this entire NLP pipeline will be GPU-accelerated using cuDF string kernels to prevent the CPU from becoming a bottleneck during data cleaning.
- **Defined the "Cold-Start" Strategy:** We will utilize the sentiment of a new user's text input to generate immediate recommendations. This ensures the system is functional for guests who are not part of the Amazon Reviews 2023 training set.
- **Refined Data Management for the RTX 2080 Ti:** Due to the 11GB VRAM limit, we will start with the "Video Games" category. We will implement column pruning and manageable chunking to ensure the 750GB dataset is processed without memory overflow.
- **Established Advanced Evaluation Metrics:** In addition to RMSE and Precision@K, we will track NDCG (Normalized Discounted Cumulative Gain) to evaluate our ranking quality. We will also benchmark Host-to-Device data transfer overhead to provide a transparent "Speedup Factor".
- **Benchmarking Plan:** We will implement a dual-pipeline (CPU vs. GPU) to showcase the performance gains of RAPIDS (cuDF/cuML) over traditional Pandas/Scikit-learn workflows.
- **Seek Guidance:** We plan to ask the professor if he prefers a production-level serving tool like NVIDIA Triton or if a standard Flask-React integration is sufficient for the capstone scope.
