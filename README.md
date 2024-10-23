# AutoEncoders for Anomaly Detection

An autoencoder is trained to compress (encode) and then reconstruct (decode) the input data. For normal data, the autoencoder learns to reconstruct the input well, minimizing reconstruction error. However, when presented with anomalous data, the reconstruction error tends to be higher, since it cannot effectively map these inputs to low-dimensional / latent representations.

![this is what an AutoEncoder looks like](images/sample_model_architecture.png)

In an anomaly detection system, a threshold is set for the reconstruction error, and anything exceeding this threshold would be flagged as an anomaly.

[Project Doc](https://docs.google.com/document/d/126OD_bMbBBT3juD-3pEXXekXodEzLmM2Tm8x-HFSCbU/edit?usp=sharing)

## Relation with IoT Devices
IoT devices can be clubbed for 'smart logistics', which is basically optimization of the supply chain between Point of Origin to warehouses in manufacturing plants. Each itemset is packed in a container, having an IoT sensor. The sensor records/transmits the following data : 
<ol>
<li> GPS coordinates </li>
<li> Acceleration </li>
</ol>

Using the following data, we can be notified of the following anomalies :
<ol>
<li> physical tampering of items (containers opened during transit) </li>
<li> container mishandling </li>
<li> overturning </li>
<li> non-optimal route selection </li>
</ol>

## Issues with ML/DL for IoT
IoT devices need to have Ultra Reliable, Low Latency Communication along with a low-power operation. This severly limits operations. We encounter the following issues :
<ol>
<li> scalability issues </li>
<li> resource limitations on IoT device </li>
<li> low power consumption requirement keeps computational power low </li>
</ol>

To counter these issues, we can make use of the following bypass : 
1. Feed metadata to a locally run ML/DL model for anomaly detection, and only send out impluses if any anomaly is detected
2. aggegate data over time, send to server periodically for analysis. Data Transfer is through UDP to cater to low-latency requirements

