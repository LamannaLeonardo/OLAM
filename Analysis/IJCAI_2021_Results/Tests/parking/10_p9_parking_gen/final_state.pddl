(define   (problem parking)
(:domain parking)
(:objects
car_0  car_1  car_2  car_3  car_4  car_5  car_6  car_7 - car
curb_0 curb_1 curb_2 curb_3 curb_4 curb_5 curb_6 curb_7 - curb
)
	(:init
			(at-curb car_3)
			(at-curb car_5)
			(at-curb car_6)
			(at-curb car_7)
			(at-curb-num car_3 curb_0)
			(at-curb-num car_5 curb_3)
			(at-curb-num car_6 curb_2)
			(at-curb-num car_7 curb_1)
			(behind-car car_0 car_7)
			(behind-car car_1 car_3)
			(behind-car car_2 car_6)
			(behind-car car_4 car_4)
			(car-clear car_0)
			(car-clear car_1)
			(car-clear car_2)
			(car-clear car_5)
			(curb-clear curb_4)
			(curb-clear curb_5)
			(curb-clear curb_6)
			(curb-clear curb_7)
	)
(:goal
(and
(at-curb-num car_0 curb_0)
(at-curb-num car_1 curb_1)
(at-curb-num car_2 curb_2)
(at-curb-num car_3 curb_3)
(at-curb-num car_4 curb_4)
(at-curb-num car_5 curb_5)
(at-curb-num car_6 curb_6)
(at-curb-num car_7 curb_7)
)
)
)


