(define (problem prob)
(:domain spanner)
(:objects
bob - man
spanner1 spanner2 spanner3 spanner4 - spanner
nut1 nut2 nut3 nut4 nut5 - nut
location1 location2 location3 location4 location5 location6 location7 location8 location9 - location
shed gate - location
)
	(:init
			(at bob shed)
			(at nut1 gate)
			(at nut2 gate)
			(at nut3 gate)
			(at nut4 gate)
			(at nut5 gate)
			(at spanner1 location4)
			(at spanner2 location2)
			(at spanner3 location5)
			(at spanner4 location1)
			(link location1 location2)
			(link location2 location3)
			(link location3 location4)
			(link location4 location5)
			(link location5 location6)
			(link location6 location7)
			(link location7 location8)
			(link location8 location9)
			(link location9 gate)
			(link shed location1)
			(loose nut1)
			(loose nut2)
			(loose nut3)
			(loose nut4)
			(loose nut5)
			(useable spanner1)
			(useable spanner2)
			(useable spanner3)
			(useable spanner4)
	)
(:goal
(and
(tightened nut1)
(tightened nut2)
(tightened nut3)
(tightened nut4)
(tightened nut5)
)))

