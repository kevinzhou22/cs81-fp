Run world:
```
rosrun stage_ros stageros PA1.world
```

Modules:
* Finder
  * Publish below communication
  * May move robot before isFound is true
* Learner
  * movement (Twist)


Communication:

Theta guaranteed [-pi, pi)
```
{
 robot: (x float, y float, theta float),
 obstacle: (x float, y float, theta float),
 isFound: true/false
 occupancyGrid: normal grid, -1 unknown, except with 200 for obstacle
}
```




