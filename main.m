% Main; Radar and Sentinel-2 fusion pipeline with machine learning
clc; clear; close all;

% Read radar data
[Ih, Qh, Iv, Qv] = read_radar_data();

% Radar data processing (derive radar moments and reflectivity beam as a grid)
[ZHmean, R_km, ReflectivityGrid, LonGrid, LatGrid] = process_radar_data(Ih, Qh, Iv, Qv);

% Read Sentinel-2 data
[sentinelWti, sentinelFcc, wtiTifFile] = read_sentinel_data();

% Feature extraction: wti, std, entropy, fcc
featureTable = extract_features(sentinelWti, sentinelFcc);

% Machine learning model (SVM) implementation for the region of interest
[satelliteImageCorrected, geoRef, lat, lon] = implement_svm_model( ...
    featureTable, sentinelWti, wtiTifFile);

% Data fusion and visualization
[fusionResultRoi, fusionResultAll] = fusion_data( ...
    LonGrid, LatGrid, ReflectivityGrid, lon, lat, satelliteImageCorrected, geoRef); 

disp("✅ Pipeline finished.");

