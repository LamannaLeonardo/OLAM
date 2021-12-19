
(define (domain ferry)
(:requirements :typing)
(:types car location)

(:predicates
		(not-eq ?x - location ?y - location)
		(at-ferry ?l - location)
		(at ?c - car ?l - location)
		(empty-ferry)
		(on ?c - car)
)


(:action sail  
:parameters (?param_1 - location ?param_2 - location)  
:precondition (and
					(at-ferry ?param_1) (not-eq ?param_1 ?param_2) (not-eq ?param_2 ?param_1)
)
:effect (and (at-ferry ?param_2) (not (at-ferry ?param_1))))

(:action board  
:parameters (?param_1 - car ?param_2 - location)  
:precondition (and
					(at ?param_1 ?param_2) (at-ferry ?param_2) (empty-ferry)
)
:effect (and (on ?param_1) (not (at ?param_1 ?param_2)) (not (empty-ferry))))

(:action debark  
:parameters (?param_1 - car ?param_2 - location)  
:precondition (and
					(at-ferry ?param_2) (on ?param_1)
)
:effect (and (at ?param_1 ?param_2) (empty-ferry) (not (on ?param_1))))
)










