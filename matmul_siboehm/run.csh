nvcu sgemm_naive
nvcu sgemm_naive_coalesced

nvcu sgemm_naive && ./sgemm_naive
nvcu sgemm_naive_coalesced && ./sgemm_naive_coalesced


profile sgemm_naive
profile sgemm_naive_coalesced