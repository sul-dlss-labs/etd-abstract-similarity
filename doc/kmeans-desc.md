## K-Means Clusters
Using the `abstracts_embeddings` matrix, the K-Means clustering approach takes
a number of clusters and
centers each cluster by calculating the Euclidean distance between the center
in each cluster with its observations.

One method to determine the number of clusters is to calculate the
**Within Cluster Sum of Squares (wcss)** for a range of cluster sizes. The
*wcss* is the sum of variance between the observations in each cluster.
Graphing the *wcss* over the number of clusters, we can pick a number that
balances the steepness of the curve verses the smoothing tail.

For more information, refer to
[K-means clustering](https://365datascience.com/k-means-clustering/)  article.
