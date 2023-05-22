define lidar ranger
(
  sensor
  (
    range [0 20]
    fov 360.0
    samples 720
  )
  # generic model properties
  color "black"
)

define rosbot position
(
  pose [ 0.0 0.0 0.0 0.0 ]
  #localization "gps"  
  localization "odom"
  #odom_error [0.0 0.00001 0.0 0.00001 0.0 0.00001]
  odom_error [ 0 0 0 0 ]

  size [ 0.2 0.2 0.2 ]
  origin [ 0.0 0.0 0.0 0.0 ]
  gui_nose 1
  drive "diff"
  color "gray"
  
  lidar(pose [ 0.0 0.0 -0.1 0.0 ])
)



