# audiosim
 

project for evaluating the performance of different approaches for quantifying song similarity 

currently using 30 second song samples provided by Spotify's API 

current planned architecture:
- local feature extraction (mp3->wav->windowing->fft->mel filterbank->mel spectral blocks) 
- processing (pca whitening, compression) 
- model (mcRBM, k-means, mRBM) 
- aggregation / high level feature extraction (feature histogram)
- distance measure (KL, cos, l^2)
