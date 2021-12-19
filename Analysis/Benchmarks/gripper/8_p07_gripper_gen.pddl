


(define (problem gripper-4-7-8)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room1 room2 room3 room4 room5 room6 room7 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 - object)
(:init
(at-robby robot1 room3)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room7)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room6)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room2)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball1 room1)
(at ball2 room6)
(at ball3 room5)
(at ball4 room3)
(at ball5 room5)
(at ball6 room7)
(at ball7 room6)
(at ball8 room1)
)
(:goal
(and
(at ball1 room3)
(at ball2 room1)
(at ball3 room2)
(at ball4 room2)
(at ball5 room5)
(at ball6 room4)
(at ball7 room1)
(at ball8 room7)
)
)
)


