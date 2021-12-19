(define (problem elevators-sequencedstrips-p4_3_1)
(:domain elevators-sequencedstrips)
(:objects
n0 n1 n2 n3 n4  - count
p0 p1 p2  - passenger
fast0  - fast-elevator
slow0-0 - slow-elevator
)
	(:init
			(above n0 n1)
			(above n0 n2)
			(above n0 n3)
			(above n0 n4)
			(above n1 n2)
			(above n1 n3)
			(above n1 n4)
			(above n2 n3)
			(above n2 n4)
			(above n3 n4)
			(boarded p2 fast0)
			(can-hold fast0 n1)
			(can-hold fast0 n2)
			(can-hold slow0-0 n1)
			(can-hold slow0-0 n2)
			(lift-at fast0 n2)
			(lift-at slow0-0 n0)
			(next n0 n1)
			(next n1 n2)
			(next n2 n3)
			(next n3 n4)
			(passenger-at p0 n2)
			(passenger-at p1 n0)
			(passengers fast0 n1)
			(passengers slow0-0 n0)
			(reachable-floor fast0 n0)
			(reachable-floor fast0 n2)
			(reachable-floor fast0 n4)
			(reachable-floor slow0-0 n0)
			(reachable-floor slow0-0 n1)
			(reachable-floor slow0-0 n2)
			(reachable-floor slow0-0 n3)
			(reachable-floor slow0-0 n4)
	)
(:goal
(and
(passenger-at p0 n3)
(passenger-at p1 n2)
(passenger-at p2 n0)
))
)


























































