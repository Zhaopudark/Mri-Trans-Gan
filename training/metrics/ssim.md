SSIM estimates covariances with weighted sums.  The default parameters
  use a biased estimate of the covariance:
  Suppose `reducer` is a weighted sum, then the mean estimators are
    $\mu_x = \sum_i w_i x_i$
    $\mu_y = \sum_i w_i y_i$,
  where w_i's are the weighted-sum weights, and covariance estimator is
    $cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)$
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    $E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y)$
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument $(1 - \sum_i w_i ^ 2)$


$cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
= \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j)$