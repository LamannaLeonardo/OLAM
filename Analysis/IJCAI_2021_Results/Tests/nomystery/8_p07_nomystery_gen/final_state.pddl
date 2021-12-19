(define (problem transport-l7-t1-p5---int100n100-m1---int100c100---s1---e0)
(:domain transport-strips)
(:objects
l0 l1 l2 l3 l4 l5 l6 - location
t0 - truck
p0 p1 p2 p3 p4 - package
level0 level1 level2 level3 level4 level5 level6 level7 level8 - fuellevel
)
	(:init
			(at p0 l3)
			(at p1 l0)
			(at p2 l1)
			(at p3 l4)
			(at p4 l3)
			(at t0 l1)
			(connected l0 l3)
			(connected l0 l6)
			(connected l1 l2)
			(connected l1 l3)
			(connected l1 l4)
			(connected l2 l1)
			(connected l2 l5)
			(connected l2 l6)
			(connected l3 l0)
			(connected l3 l1)
			(connected l4 l1)
			(connected l5 l2)
			(connected l6 l0)
			(connected l6 l2)
			(fuel t0 level0)
			(fuelcost level1 l0 l3)
			(fuelcost level1 l0 l6)
			(fuelcost level1 l1 l2)
			(fuelcost level1 l1 l3)
			(fuelcost level1 l1 l4)
			(fuelcost level1 l2 l1)
			(fuelcost level1 l2 l5)
			(fuelcost level1 l2 l6)
			(fuelcost level1 l3 l0)
			(fuelcost level1 l3 l1)
			(fuelcost level1 l4 l1)
			(fuelcost level1 l5 l2)
			(fuelcost level1 l6 l0)
			(fuelcost level1 l6 l2)
			(sum level0 level0 level0)
			(sum level0 level1 level1)
			(sum level0 level2 level2)
			(sum level0 level3 level3)
			(sum level0 level4 level4)
			(sum level0 level5 level5)
			(sum level0 level6 level6)
			(sum level0 level7 level7)
			(sum level0 level8 level8)
			(sum level1 level0 level1)
			(sum level1 level1 level2)
			(sum level1 level2 level3)
			(sum level1 level3 level4)
			(sum level1 level4 level5)
			(sum level1 level5 level6)
			(sum level1 level6 level7)
			(sum level1 level7 level8)
			(sum level2 level0 level2)
			(sum level2 level1 level3)
			(sum level2 level2 level4)
			(sum level2 level3 level5)
			(sum level2 level4 level6)
			(sum level2 level5 level7)
			(sum level2 level6 level8)
			(sum level3 level0 level3)
			(sum level3 level1 level4)
			(sum level3 level2 level5)
			(sum level3 level3 level6)
			(sum level3 level4 level7)
			(sum level3 level5 level8)
			(sum level4 level0 level4)
			(sum level4 level1 level5)
			(sum level4 level2 level6)
			(sum level4 level3 level7)
			(sum level4 level4 level8)
			(sum level5 level0 level5)
			(sum level5 level1 level6)
			(sum level5 level2 level7)
			(sum level5 level3 level8)
			(sum level6 level0 level6)
			(sum level6 level1 level7)
			(sum level6 level2 level8)
			(sum level7 level0 level7)
			(sum level7 level1 level8)
			(sum level8 level0 level8)
	)
(:goal
(and
(at p0 l5)
(at p1 l1)
(at p2 l2)
(at p3 l1)
(at p4 l2)
)
)
)









