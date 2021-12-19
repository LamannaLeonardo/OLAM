(define (problem ferry-l4-c1)
(:domain ferry)
(:objects l0 l1 l2 l3 - location
c0 - car
)
	(:init
			(at-ferry l1)
			(not-eq l0 l1)
			(not-eq l0 l2)
			(not-eq l0 l3)
			(not-eq l1 l0)
			(not-eq l1 l2)
			(not-eq l1 l3)
			(not-eq l2 l0)
			(not-eq l2 l1)
			(not-eq l2 l3)
			(not-eq l3 l0)
			(not-eq l3 l1)
			(not-eq l3 l2)
			(on c0)
	)
(:goal
(and
(at c0 l1)
)
)
)








