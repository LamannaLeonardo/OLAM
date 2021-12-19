(define   (problem parking)
(:domain parking)
(:objects
car_0  car_1 - car
curb_0 curb_1 - curb
)
	(:init
			(behind-car car_0 car_0)
			(behind-car car_1 car_1)
			(curb-clear curb_0)
			(curb-clear curb_1)
	)
(:goal
(and
(at-curb-num car_0 curb_0)
(at-curb-num car_1 curb_1)
)
)
)




