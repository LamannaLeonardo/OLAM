


(define (problem ferry-l5-c2)
(:domain ferry)
(:objects l0 l1 l2 l3 l4 - location
          c0 c1 - car
)
(:init
(not-eq l0 l1)
(not-eq l1 l0)
(not-eq l0 l2)
(not-eq l2 l0)
(not-eq l0 l3)
(not-eq l3 l0)
(not-eq l0 l4)
(not-eq l4 l0)
(not-eq l1 l2)
(not-eq l2 l1)
(not-eq l1 l3)
(not-eq l3 l1)
(not-eq l1 l4)
(not-eq l4 l1)
(not-eq l2 l3)
(not-eq l3 l2)
(not-eq l2 l4)
(not-eq l4 l2)
(not-eq l3 l4)
(not-eq l4 l3)
(empty-ferry)
(at c0 l0)
(at c1 l1)
(at-ferry l4)
)
(:goal
(and
(at c0 l4)
(at c1 l1)
)
)
)


