(define (problem transport-l3-t1-p1---int100n100-m1---int100c100---s1---e0)
(:domain transport-strips)

(:objects
l0 l1 l2 - location
t0 - truck
p0 - package
level0 level1 level2 - fuellevel
)

(:init
(sum level0 level0 level0)
(sum level0 level1 level1)
(sum level0 level2 level2)
(sum level1 level0 level1)
(sum level1 level1 level2)
(sum level2 level0 level2)

(connected l0 l1)
(fuelcost level1 l0 l1)
(connected l0 l2)
(fuelcost level1 l0 l2)
(connected l1 l0)
(fuelcost level1 l1 l0)
(connected l1 l2)
(fuelcost level1 l1 l2)
(connected l2 l0)
(fuelcost level1 l2 l0)
(connected l2 l1)
(fuelcost level1 l2 l1)

(at t0 l0)
(fuel t0 level2)
(= (total-cost) 0)

(at p0 l2)
)

(:goal
(and
(at p0 l0)
)
)
(:metric minimize (total-cost))
)
