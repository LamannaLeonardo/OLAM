


(define (problem gripper-5-8-10)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 robot5 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 rgripper5 lgripper5 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 ball8 ball9 ball10 - object)
(:init
(at-robby robot1 room2)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room2)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room5)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room8)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at-robby robot5 room1)
(free robot5 rgripper5)
(free robot5 lgripper5)
(at ball1 room6)
(at ball2 room2)
(at ball3 room3)
(at ball4 room6)
(at ball5 room8)
(at ball6 room3)
(at ball7 room7)
(at ball8 room5)
(at ball9 room7)
(at ball10 room6)
)
(:goal
(and
(at ball1 room5)
(at ball2 room4)
(at ball3 room7)
(at ball4 room5)
(at ball5 room6)
(at ball6 room8)
(at ball7 room1)
(at ball8 room8)
(at ball9 room8)
(at ball10 room6)
)
)
)


