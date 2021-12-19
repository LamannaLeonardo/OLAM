(define (problem prob)
(:domain spanner)
(:objects
bob - man
spanner1 - spanner
nut1 nut2 - nut
location1 location2 location3 - location
shed gate - location
)
	(:init
			(at bob gate)
			(at nut1 gate)
			(at nut2 gate)
			(carrying bob spanner1)
			(link location1 location2)
			(link location2 location3)
			(link location3 gate)
			(link shed location1)
			(loose nut2)
			(tightened nut1)
	)
(:goal
(and
(tightened nut1)
(tightened nut2)
)))







