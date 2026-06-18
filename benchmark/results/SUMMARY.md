## Benchmark result tables

??? info "Encoding at matched PSNR (frames/s and compression at equal image quality, interpolated from the sweep; x_psnr_matched=False means the encoder's sweep did not reach the target and the nearest point is shown)"

    | workload | backend | x_target_psnr | x_psnr_db | x_psnr_matched | metric_main | x_compression_ratio | x_file_size_mb |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | enc_hd | opencv | 34.6 | 36.1 | False | 260.1 | 7.4 | 4.99 |
    | enc_hd | pvio_cpu | 34.6 | 34.6 | True | 95.1 | 15.4 | 2.41 |
    | enc_hd | pvio_gpu | 34.6 | 34.6 | True | 212.6 | 43.8 | 0.85 |
    | enc_hd | pyav | 34.6 | 34.6 | True | 40.0 | 4.1 | 8.93 |
    | enc_sd | opencv | 34.61 | 36.09 | False | 530.6 | 7.2 | 2.29 |
    | enc_sd | pvio_cpu | 34.61 | 34.61 | True | 153.8 | 14.2 | 1.15 |
    | enc_sd | pvio_gpu | 34.61 | 34.61 | True | 322.1 | 41.8 | 0.39 |
    | enc_sd | pyav | 34.61 | 34.61 | True | 72.1 | 4.2 | 3.96 |

??? info "Encoding Pareto sweep (throughput vs compression per quality level)"

    | workload | backend | x_quality_param | metric_main | x_compression_ratio | x_psnr_db |
    | --- | --- | --- | --- | --- | --- |
    | enc_hd | opencv | 51.0 | 258.698 | 7.4 | 36.1 |
    | enc_hd | opencv | 55.0 | 241.261 | 7.4 | 36.1 |
    | enc_hd | opencv | 59.0 | 247.069 | 7.4 | 36.1 |
    | enc_hd | opencv | 63.0 | 253.897 | 7.4 | 36.1 |
    | enc_hd | opencv | 67.0 | 260.121 | 7.4 | 36.1 |
    | enc_hd | pvio_cpu | 17.0 | 42.234 | 2.1 | 35.5 |
    | enc_hd | pvio_cpu | 19.0 | 56.413 | 4.4 | 35.04 |
    | enc_hd | pvio_cpu | 21.0 | 77.639 | 9.8 | 34.85 |
    | enc_hd | pvio_cpu | 23.0 | 105.692 | 19.5 | 34.47 |
    | enc_hd | pvio_cpu | 25.0 | 152.335 | 46.2 | 34.01 |
    | enc_hd | pvio_gpu | 17.0 | 208.889 | 6.1 | 35.33 |
    | enc_hd | pvio_gpu | 19.0 | 204.35 | 14.0 | 35.14 |
    | enc_hd | pvio_gpu | 21.0 | 212.7 | 38.7 | 34.75 |
    | enc_hd | pvio_gpu | 23.0 | 212.342 | 53.8 | 34.35 |
    | enc_hd | pvio_gpu | 25.0 | 210.236 | 80.3 | 34.18 |
    | enc_hd | pyav | 17.0 | 30.977 | 2.1 | 35.02 |
    | enc_hd | pyav | 19.0 | 40.755 | 4.3 | 34.57 |
    | enc_hd | pyav | 21.0 | 57.098 | 9.8 | 34.37 |
    | enc_hd | pyav | 23.0 | 75.433 | 19.6 | 34.1 |
    | enc_hd | pyav | 25.0 | 103.73 | 46.3 | 33.96 |
    | enc_sd | opencv | 51.0 | 556.54 | 7.2 | 36.09 |
    | enc_sd | opencv | 55.0 | 502.394 | 7.2 | 36.09 |
    | enc_sd | opencv | 59.0 | 472.437 | 7.2 | 36.09 |
    | enc_sd | opencv | 63.0 | 514.863 | 7.2 | 36.09 |
    | enc_sd | opencv | 67.0 | 530.643 | 7.2 | 36.09 |
    | enc_sd | pvio_cpu | 17.0 | 80.225 | 2.1 | 35.49 |
    | enc_sd | pvio_cpu | 19.0 | 103.236 | 4.6 | 35.03 |
    | enc_sd | pvio_cpu | 21.0 | 131.117 | 10.5 | 34.8 |
    | enc_sd | pvio_cpu | 23.0 | 189.809 | 21.2 | 34.36 |
    | enc_sd | pvio_cpu | 25.0 | 277.67 | 49.4 | 33.98 |
    | enc_sd | pvio_gpu | 17.0 | 311.892 | 6.0 | 35.36 |
    | enc_sd | pvio_gpu | 19.0 | 315.053 | 14.0 | 35.17 |
    | enc_sd | pvio_gpu | 21.0 | 330.707 | 36.6 | 34.76 |
    | enc_sd | pvio_gpu | 23.0 | 309.921 | 50.8 | 34.39 |
    | enc_sd | pvio_gpu | 25.0 | 303.604 | 74.0 | 34.21 |
    | enc_sd | pyav | 17.0 | 58.296 | 2.1 | 35.01 |
    | enc_sd | pyav | 19.0 | 74.396 | 4.6 | 34.55 |
    | enc_sd | pyav | 21.0 | 108.504 | 10.5 | 34.32 |
    | enc_sd | pyav | 23.0 | 136.823 | 21.1 | 34.03 |
    | enc_sd | pyav | 25.0 | 187.664 | 48.2 | 33.75 |

??? info "Random access (frames/s, higher better)"

    | workload | opencv_cpu | pvio_cpu | pvio_gpu | pyav_cpu | torchcodec_cpu | torchcodec_cuda |
    | --- | --- | --- | --- | --- | --- | --- |
    | fhd_h264 | 3.0 | 23.0 | 133.0 | 6.0 | 34.0 | 137.0 |
    | hd_h264 | 7.0 | 49.0 | 232.0 | 13.0 | 75.0 | 257.0 |
    | sd_h264 | 17.0 | 104.0 | 397.0 | 29.0 | 152.0 | 490.0 |

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
    | fhd_h264 | 97.0 | 65.0 | 553.0 | 109.0 | 100.0 | 563.0 |
    | hd_h264 | 233.0 | 141.0 | 1106.0 | 233.0 | 212.0 | 1104.0 |
    | sd_h264 | 530.0 | 434.0 | 2105.0 | 586.0 | 491.0 | 2100.0 |

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
