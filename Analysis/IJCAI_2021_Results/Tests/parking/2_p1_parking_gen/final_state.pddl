(define   (problem parking)
(:domain parking)
(:objects
car_0  car_1  car_2 - car
curb_0 curb_1 curb_2 - curb
)
	(:init
			(at-curb car_2)
			(at-curb-num car_2 curb_0)
			(behind-car car_0 car_0)
			(behind-car car_1 car_1)
			(car-clear car_2)
			(curb-clear curb_1)
			(curb-clear curb_2)
	)
(:goal
(and
(at-curb-num car_0 curb_0)
(at-curb-num car_1 curb_1)
(at-curb-num car_2 curb_2)
)
)
)








