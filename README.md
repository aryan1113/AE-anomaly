# AutoEncoders for Anomaly Detection

An autoencoder is trained to compress (encode) and then reconstruct (decode) the input data. For normal data, the autoencoder learns to reconstruct the input well, minimizing reconstruction error. However, when presented with anomalous data, the reconstruction error tends to be higher, since it cannot effectively map these inputs to low-dimensional / latent representations.

In an anomaly detection system, a threshold is set for the reconstruction error, and anything exceeding this threshold would be flagged as an anomaly.

[Project Doc](https://docs.google.com/document/d/126OD_bMbBBT3juD-3pEXXekXodEzLmM2Tm8x-HFSCbU/edit?usp=sharing){:target = "_blank"}

