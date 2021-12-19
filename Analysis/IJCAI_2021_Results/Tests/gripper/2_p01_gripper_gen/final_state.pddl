(define (problem gripper-1-4-2)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 - room
ball1 ball2 - object)
	(:init
			(at ball2 room4)
			(at-robby robot1 room4)
			(carry robot1 ball1 rgripper1)
			(free robot1 lgripper1)
	)
(:goal
(and
(at ball1 room4)
(at ball2 room3)
)
)
)



