(define (problem mixed-f2-p2-u0-v0-d0-a0-n0-a0-b0-n0-f0)
(:domain miconic)
(:objects p0 p1 - passenger
f0 f1 - floor)
	(:init
			(above f0 f1)
			(destin p0 f0)
			(destin p1 f0)
			(lift-at f0)
			(origin p0 f1)
			(origin p1 f1)
	)
(:goal (and
(served p0)
(served p1)
))
)

