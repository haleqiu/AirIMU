## SubT-MRS
Subt-MRS dataset is a multi-robot, multi-modal and multi-degraded dataset, an extremely challenging real-world dataset designed to push SLAM towards all-weather environments.
In our experiments, we selected the multi-agent UGV datasets from the LiDAR-inertial track, which is collectedin the DARPA Subterranean (SubT) Challenge’s Final Event and Urban Circuit Event. 

|Challenge|Robot|Subseq Num| Environment          |Trajectory | Duration(s) | 
|------|----|------|-------------| -----------|-----------|
|Final Challenge|UGV1|4|Infrastructure| 563m   | 1641s        | 
|Final Challenge|UGV2|9|Tunnel| 634m      |3363s|
|Final Challenge|UGV3|6|Tunnel and Cave|  757m    |1717s|
|Urban Challenge|UGV1|2|Factory|154m   |   491s      | 
|Urban Challenge|UGV2|7|Factory  | 1726m      |3115s|


#### Dataset File Structure
```
  SubT-MRS
  ├── Final_Challenge
  │   ├── UGV1
  │   │   ├── infrastructure_corridor_1
  │   │   │   ├── ground_truth
  │   │   │   │   └── ground_truth_imu.csv
  │   │   │   └── imu_data
  │   │   │   │   └── imu_data.csv          
  │   │   ├── infrastructure_corridor_2
  │   │   ├── warehouse_1
  │   │   └── warehouse_2
  │   ├── UGV2
  │   │   ├── tunnel_corridor_1
  │   │   ├── tunnel_corridor_2
  │   │   ├── tunnel_corridor_3
  │   │   ├── tunnel_intermittence
  │   │   ├── tunnel_room
  │   │   ├── tunnel_stop_1
  │   │   ├── tunnel_stop_2
  │   │   ├── tunnel_stop_3
  │   │   └── tunnel_stop_4
  │   └── UGV3
  │       ├── cave_corridor_1
  │       ├── cave_corridor_2
  │       ├── cave_corridor_3
  │       ├── cave_corridor_4
  │       ├── tunnel_corridor_4
  │       └── tunnel_corridor_5
  └── Urban_Challenge
      ├── UGV1
      │   ├── factory_1
      │   │   ├── ground_truth
      │   │   │   └── ground_truth_imu.csv
      │   │   └── imu_data
      │   │   │   └── imu_data.csv          
      │   └── factory_2
      └── UGV2
          ├── factory_3
          ├── factory_4
          ├── factory_5
          ├── factory_6
          ├── factory_7
          ├── factory_8
          └── factory_9
```
#### Payload Information

| Component       | Type                        | Rate    | Characteristics            |
| --------------- | --------------------------- | ------- | -------------------------- |
| IMU             | Epson M-G365                | 200Hz   | Time Synchronization center|

##### IMU Calibration
Used the M-G365 inertial sensor1 on our platform and calibrate it to reduce bias instability and drift. Employed an Allan variance[https://ieeexplore.ieee.org/document/4404126] based tool2 to estimate the white noise angle random walk and bias instability for both the gyroscope and accelerometer data.
