# MeanFeaureReconstructionError
See https://arxiv.org/pdf/1603.08155.pdf, consider $j$ is a convolutional layer and ${\phi}_{j}(x)$ is a feature map of shape $C_{j}\times H_{j}\times W_{j}$, then we got "Feature Reconstruction Loss" for a single layer $j$ as:

$$
l^{\phi,j}_{feat}(\hat{y},y)=\frac{1}{C_{j}H_{j}W_{j}}\left \| {\phi}_{j}(\hat{y})-{\phi}_{j}(y) \right \|^{2}_{2}
$$

when application, consider ${\phi}_{j}(x)$ is a feature map of N dims shape $[B,D_{2},D_{3},...,D_{N}]$, the  above formular can be rewrite as:

$$
l^{\phi,j}_{feat}(\hat{y},y)=\frac{1}{\prod_{i=2}^ND_{i}}\left \| {\phi}_{j}(\hat{y})-{\phi}_{j}(y) \right \|^{2}_{2}
$$

where the $\left \| \cdot \right \|^{2}_{2}$ calculates norm from the dimension $D_{2}$ to $D_{N}$, except the batch dimesion. So, output shape is $[B]$. This calculation behavior is very important, beacuse, theoretically, "Feature Reconstruction Loss" considers only one sample's feature difference from its reconstructed result. If consider two or more samples as a same time, there is no guarantee that model can reconstruct each one well, i.e., minimizing the mean "Feature Reconstruction Loss" on two or more samples not necessarily leads to minimizing each sample's "Feature Reconstruction Loss". So, the rewritten formular maintains the batch dimesion. A user can customize whether to calculate the mean over batch dimension.

# MeanStyleReconstructionError

See https://arxiv.org/pdf/1603.08155.pdf, consider $j$ is a convolutional layer and ${\phi}_{j}(x)$ is a feature map of shape $C_{j}\times H_{j}\times W_{j}$, then we got its 'Gram Matrix' $G^{\phi}_{j}(x)$ as a $C_j\times C_j$ matrix whos elements are give by:

$$
    G^{\phi}_{j}(x)_{c,c'}=\frac{1}{C_{j}H_{j}W_{j}}\sum_{h=1}^{H_j}\sum_{w=1}^{W_j}{\phi}_{j}(x)_{c,h,w}{\phi}_{j}(x)_{c',h,w}
$$

where, $c$ or $c'$ represent the channel index of ${\phi}_{j}(x)$. $G^{\phi}_{j}(x)$ is proportional to the uncentered covariance of the Cj-dimensional features, treating each grid location as an independent sample. It thus captures information about which features tend to activate together. So the information represents general style. 

when practice, consider ${\phi}_{j}(x)$ is a feature map of N dims shape $[B,D_{2},D_{3},...,D_{N-1},C]$ or $[B,C,D_{2},D_{3},...,D_{N-1}]$, the above formular can be rewrite as:

$$
    G^{\phi}_{j}(x)_{c,c'}=\frac{1}{C\prod_{i=2}^{N-1}D_{i}}\sum_{d_2=1}^{D_2}\sum_{d_3=1}^{D_3}\cdots \sum_{d_{N-1}=1}^{D_{N-1}} {\phi}_{j}(x)_{c,d_2,d_3,\ldots,d_{N-1}}{\phi}_{j}(x)_{c',d_2,d_3,\ldots,d_{N-1}}
$$


Then, the "Style Reconstruction Loss" can be formulated as

$$
l^{\phi,j}_{style}(\hat{y},y)=\left \| G^{\phi}_{j}(\hat{y})-G^{\phi}_{j}(y) \right \|^{2}_{F}
$$




