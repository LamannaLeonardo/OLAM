
(define (domain typed-sokoban)
(:requirements :typing)
(:types loc dir box)
(:predicates 
             (at-robot ?l - loc)
             (at ?o - box ?l - loc)
             (adjacent ?l1 - loc ?l2 - loc ?d - dir) 
             (clear ?l - loc)
)



(:action move
:parameters (?param_1 - loc ?param_2 - loc ?param_3 - dir)
:precondition (and
		(adjacent ?param_1 ?param_2 ?param_3) (at-robot ?param_1) (clear ?param_2)
)
:effect (and (at-robot ?param_2) (not (at-robot ?param_1))))

(:action push
:parameters (?param_1 - loc ?param_2 - loc ?param_3 - loc ?param_4 - dir ?param_5 - box)
:precondition (and
		(adjacent ?param_1 ?param_2 ?param_4) (adjacent ?param_2 ?param_3 ?param_4) (at ?param_5 ?param_2) (at-robot ?param_1) (clear ?param_3)
)
:effect (and (clear ?param_2) (at ?param_5 ?param_3) (at-robot ?param_2) (not (clear ?param_3)) (not (at ?param_5 ?param_2)) (not (at-robot ?param_1))))
)





