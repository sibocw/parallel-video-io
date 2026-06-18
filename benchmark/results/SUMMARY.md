## Benchmark result tables

??? info "Encoding at matched PSNR (frames/s and compression at equal image quality, interpolated from the sweep; x_psnr_matched=False means the encoder's sweep did not reach the target and the nearest point is shown)"

    | workload | backend | x_target_psnr | x_psnr_db | x_psnr_matched | metric_main | x_compression_ratio | x_file_size_mb |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | enc_hd | opencv | 34.7 | 36.1 | False | 268.1 | 7.4 | 4.99 |
    | enc_hd | pvio_cpu | 34.7 | 34.7 | True | 93.4 | 14.9 | 2.48 |
    | enc_hd | pvio_gpu | 34.7 | 34.7 | True | 214.8 | 39.3 | 0.94 |
    | enc_hd | pyav | 34.7 | 34.7 | True | 49.2 | 6.8 | 5.45 |
    | enc_sd | opencv | 34.61 | 36.08 | False | 590.3 | 7.1 | 2.3 |
    | enc_sd | pvio_cpu | 34.61 | 34.61 | True | 165.9 | 14.1 | 1.17 |
    | enc_sd | pvio_gpu | 34.61 | 34.61 | True | 332.7 | 43.2 | 0.38 |
    | enc_sd | pyav | 34.61 | 34.61 | True | 75.9 | 4.1 | 3.96 |

??? info "Encoding Pareto sweep (throughput vs compression per quality level)"

    | workload | backend | x_quality_param | metric_main | x_compression_ratio | x_psnr_db |
    | --- | --- | --- | --- | --- | --- |
    | enc_hd | opencv | 51.0 | 266.315 | 7.4 | 36.1 |
    | enc_hd | opencv | 55.0 | 264.555 | 7.4 | 36.1 |
    | enc_hd | opencv | 59.0 | 236.294 | 7.4 | 36.1 |
    | enc_hd | opencv | 63.0 | 257.014 | 7.4 | 36.1 |
    | enc_hd | opencv | 67.0 | 268.066 | 7.4 | 36.1 |
    | enc_hd | pvio_cpu | 17.0 | 42.393 | 2.1 | 35.64 |
    | enc_hd | pvio_cpu | 19.0 | 55.568 | 4.4 | 35.18 |
    | enc_hd | pvio_cpu | 21.0 | 77.826 | 9.8 | 34.94 |
    | enc_hd | pvio_cpu | 23.0 | 106.237 | 20.0 | 34.54 |
    | enc_hd | pvio_cpu | 25.0 | 159.0 | 48.2 | 34.11 |
    | enc_hd | pvio_gpu | 17.0 | 210.942 | 6.1 | 35.34 |
    | enc_hd | pvio_gpu | 19.0 | 214.439 | 14.1 | 35.15 |
    | enc_hd | pvio_gpu | 21.0 | 215.117 | 38.5 | 34.73 |
    | enc_hd | pvio_gpu | 23.0 | 210.149 | 53.2 | 34.34 |
    | enc_hd | pvio_gpu | 25.0 | 213.12 | 77.8 | 34.16 |
    | enc_hd | pyav | 17.0 | 31.007 | 2.1 | 35.25 |
    | enc_hd | pyav | 19.0 | 41.594 | 4.4 | 34.84 |
    | enc_hd | pyav | 21.0 | 56.799 | 9.9 | 34.59 |
    | enc_hd | pyav | 23.0 | 76.198 | 20.0 | 34.28 |
    | enc_hd | pyav | 25.0 | 105.748 | 48.3 | 34.08 |
    | enc_sd | opencv | 51.0 | 585.761 | 7.1 | 36.08 |
    | enc_sd | opencv | 55.0 | 573.155 | 7.1 | 36.08 |
    | enc_sd | opencv | 59.0 | 585.975 | 7.1 | 36.08 |
    | enc_sd | opencv | 63.0 | 574.135 | 7.1 | 36.08 |
    | enc_sd | opencv | 67.0 | 590.283 | 7.1 | 36.08 |
    | enc_sd | pvio_cpu | 17.0 | 82.001 | 2.1 | 35.49 |
    | enc_sd | pvio_cpu | 19.0 | 105.573 | 4.6 | 35.04 |
    | enc_sd | pvio_cpu | 21.0 | 144.804 | 10.3 | 34.82 |
    | enc_sd | pvio_cpu | 23.0 | 197.292 | 20.9 | 34.33 |
    | enc_sd | pvio_cpu | 25.0 | 291.479 | 48.9 | 33.91 |
    | enc_sd | pvio_gpu | 17.0 | 340.62 | 6.1 | 35.38 |
    | enc_sd | pvio_gpu | 19.0 | 336.439 | 14.3 | 35.18 |
    | enc_sd | pvio_gpu | 21.0 | 335.39 | 37.4 | 34.76 |
    | enc_sd | pvio_gpu | 23.0 | 329.102 | 52.2 | 34.4 |
    | enc_sd | pvio_gpu | 25.0 | 337.957 | 75.6 | 34.2 |
    | enc_sd | pyav | 17.0 | 58.733 | 2.1 | 35.01 |
    | enc_sd | pyav | 19.0 | 78.548 | 4.5 | 34.55 |
    | enc_sd | pyav | 21.0 | 110.983 | 10.3 | 34.32 |
    | enc_sd | pyav | 23.0 | 143.367 | 20.8 | 34.05 |
    | enc_sd | pyav | 25.0 | 203.486 | 48.0 | 33.78 |

??? info "Random access (frames/s, higher better)"

    | workload | opencv_cpu | pvio_cpu | pvio_gpu | pyav_cpu | torchcodec_cpu | torchcodec_cuda |
    | --- | --- | --- | --- | --- | --- | --- |
    | fhd_h264 | 3.0 | 22.0 | 135.0 | 6.0 | 35.0 | 139.0 |
    | hd_h264 | 8.0 | 49.0 | 262.0 | 14.0 | 81.0 | 291.0 |
    | sd_h264 | 17.0 | 121.0 | 498.0 | 32.0 | 176.0 | 554.0 |

??? info "Seek correctness (all videos)"

    | backend | x_seek_correct |
    | --- | --- |
    | opencv_cpu | True |
    | pvio_cpu | True |
    | pvio_gpu | True |
    | pyav_cpu | True |
    | torchcodec_cpu | True |
    | torchcodec_cuda | True |

??? info "Sequential access (frames/s, higher better)"

    | workload | opencv_cpu | pvio_cpu | pvio_gpu | pyav_cpu | torchcodec_cpu | torchcodec_cuda |
    | --- | --- | --- | --- | --- | --- | --- |
    | fhd_h264 | 102.0 | 67.0 | 554.0 | 112.0 | 97.0 | 554.0 |
    | hd_h264 | 245.0 | 183.0 | 1152.0 | 259.0 | 240.0 | 1169.0 |
    | sd_h264 | 545.0 | 461.0 | 2137.0 | 601.0 | 514.0 | 2133.0 |

??? info "Skipped / errored"

    | task | backend | workload | error |
    | --- | --- | --- | --- |
    | random | decord_cpu | sd_h264 | import failed: No module named 'decord' |
    | random | decord_cpu | hd_h264 | import failed: No module named 'decord' |
    | random | decord_cpu | fhd_h264 | import failed: No module named 'decord' |
    | sequential | dali_gpu | sd_h264 | ImportError: cannot import name 'fn' from 'nvidia.dali' (unknown location) |
    | sequential | decord_cpu | sd_h264 | import failed: No module named 'decord' |
    | sequential | dali_gpu | hd_h264 | ImportError: cannot import name 'fn' from 'nvidia.dali' (unknown location) |
    | sequential | decord_cpu | hd_h264 | import failed: No module named 'decord' |
    | sequential | dali_gpu | fhd_h264 | ImportError: cannot import name 'fn' from 'nvidia.dali' (unknown location) |
    | sequential | decord_cpu | fhd_h264 | import failed: No module named 'decord' |
