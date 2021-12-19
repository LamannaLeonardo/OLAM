(define (problem transport-l4-t1-p1---int100n100-m1---int100c100---s1---e0)
(:domain transport-strips)
(:objects
l0 l1 l2 l3 - location
t0 - truck
p0 - package
level0 level1 level2 level3 - fuellevel
)
	(:init
			(at p0 l2)
			(at t0 l2)
			(connected l0 l2)
			(connected l0 l3)
			(connected l1 l2)
			(connected l1 l3)
			(connected l2 l0)
			(connected l2 l1)
			(connected l3 l0)
			(connected l3 l1)
			(fuel t0 level0)
			(fuelcost level1 l0 l2)
			(fuelcost level1 l0 l3)
			(fuelcost level1 l1 l2)
			(fuelcost level1 l1 l3)
			(fuelcost level1 l2 l0)
			(fuelcost level1 l2 l1)
			(fuelcost level1 l3 l0)
			(fuelcost level1 l3 l1)
			(sum level0 level0 level0)
			(sum level0 level1 level1)
			(sum level0 level2 level2)
			(sum level0 level3 level3)
			(sum level1 level0 level1)
			(sum level1 level1 level2)
			(sum level1 level2 level3)
			(sum level2 level0 level2)
			(sum level2 level1 level3)
			(sum level3 level0 level3)
	)
(:goal
(and
(at p0 l3)
)
)
)




