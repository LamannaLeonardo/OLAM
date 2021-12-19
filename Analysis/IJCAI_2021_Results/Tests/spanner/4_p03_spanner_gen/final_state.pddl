(define (problem prob)
(:domain spanner)
(:objects
bob - man
spanner1 spanner2 - spanner
nut1 nut2 nut3 - nut
location1 location2 location3 location4 location5 location6 - location
shed gate - location
)
	(:init
			(at bob shed)
			(at nut1 gate)
			(at nut2 gate)
			(at nut3 gate)
			(at spanner1 location3)
			(at spanner2 location6)
			(link location1 location2)
			(link location2 location3)
			(link location3 location4)
			(link location4 location5)
			(link location5 location6)
			(link location6 gate)
			(link shed location1)
			(loose nut1)
			(loose nut2)
			(loose nut3)
			(useable spanner1)
			(useable spanner2)
	)
(:goal
(and
(tightened nut1)
(tightened nut2)
(tightened nut3)
)))

