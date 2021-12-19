(define (problem prob)
(:domain spanner)
(:objects
bob - man
spanner1 spanner2 - spanner
nut1 nut2 nut3 - nut
location1 location2 location3 location4 location5 - location
shed gate - location
)
	(:init
			(at bob gate)
			(at nut1 gate)
			(at nut2 gate)
			(at nut3 gate)
			(carrying bob spanner1)
			(carrying bob spanner2)
			(link location1 location2)
			(link location2 location3)
			(link location3 location4)
			(link location4 location5)
			(link location5 gate)
			(link shed location1)
			(loose nut1)
			(loose nut2)
			(tightened nut3)
			(useable spanner2)
	)
(:goal
(and
(tightened nut1)
(tightened nut2)
(tightened nut3)
)))










