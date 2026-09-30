# Synthetic benchmark summary

## Overall (all sweeps)

| method   |   pairs |   success_rate |   median_gt_rmse_px |   median_cp_rmse_px |   median_inliers |   median_coverage |   median_runtime_s |
|:---------|--------:|---------------:|--------------------:|--------------------:|-----------------:|------------------:|-------------------:|
| akaze    |      69 |          0.696 |               0.007 |               0.049 |            570   |             0.969 |              0.923 |
| rootsift |      69 |          0.783 |               0.01  |               0.049 |            567.5 |             1     |              1.48  |
| sift     |      69 |          0.768 |               0.009 |               0.049 |            563   |             0.992 |              1.523 |

## sun_azimuth: success rate (GT RMSE < 1 px)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   0 |       1 |          1 |      1 |
|  15 |       1 |          1 |      1 |
|  30 |       1 |          1 |      1 |
|  45 |       0 |          0 |      0 |
|  60 |       0 |          0 |      0 |
|  90 |       0 |          0 |      0 |
| 120 |       0 |          0 |      0 |
| 180 |       0 |          0 |      0 |

## sun_azimuth: median GT RMSE (px, successful runs)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   0 |   0.006 |      0.005 |  0.005 |
|  15 |   0.006 |      0.009 |  0.01  |
|  30 |   0.023 |      0.06  |  0.075 |

## sun_elevation: success rate (GT RMSE < 1 px)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   5 |       1 |          1 |      1 |
|  10 |       1 |          1 |      1 |
|  15 |       1 |          1 |      1 |
|  20 |       1 |          1 |      1 |
|  30 |       1 |          1 |      1 |
|  45 |       1 |          1 |      1 |
|  60 |       1 |          1 |      1 |

## sun_elevation: median GT RMSE (px, successful runs)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   5 |   0.196 |      0.165 |  0.165 |
|  10 |   0.082 |      0.068 |  0.067 |
|  15 |   0.024 |      0.025 |  0.023 |
|  20 |   0.01  |      0.01  |  0.01  |
|  30 |   0.006 |      0.005 |  0.005 |
|  45 |   0.006 |      0.006 |  0.006 |
|  60 |   0.006 |      0.008 |  0.007 |

## scale_hint: success rate (GT RMSE < 1 px)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   1 |    1    |          1 |      1 |
|   2 |    1    |          1 |      1 |
|   4 |    1    |          1 |      1 |
|   8 |    0.33 |          1 |      1 |

## scale_hint: median GT RMSE (px, successful runs)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   1 |   0.005 |      0.005 |  0.005 |
|   2 |   0.006 |      0.005 |  0.005 |
|   4 |   0.024 |      0.012 |  0.009 |
|   8 |   0.486 |      0.053 |  0.06  |

## scale_nohint: success rate (GT RMSE < 1 px)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   1 |    1    |          1 |   1    |
|   2 |    1    |          1 |   1    |
|   4 |    0.67 |          1 |   1    |
|   8 |    0    |          1 |   0.67 |

## scale_nohint: median GT RMSE (px, successful runs)

|   x |   akaze |   rootsift |   sift |
|----:|--------:|-----------:|-------:|
|   1 |   0.005 |      0.005 |  0.005 |
|   2 |   0.006 |      0.006 |  0.006 |
|   4 |   0.248 |      0.032 |  0.032 |
|   8 | nan     |      0.199 |  0.279 |
