(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 - room
ball1 - object)
	(:init
			(at ball1 room3)
			(at-robby robot1 room3)
			(free robot1 lgripper1)
			(free robot1 rgripper1)
	)
(:goal
(and
(at ball1 room2)
)
)
)




