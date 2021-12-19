
(define (domain miconic)
(:requirements :typing)
(:types floor passenger)

(:predicates
		(origin ?person - passenger ?floor - floor)
		(destin ?person - passenger ?floor - floor)
		(above ?floor1 - floor ?floor2 - floor)
		(boarded ?person - passenger)
		(served ?person - passenger)
		(lift-at ?floor - floor))


(:action board  
:parameters (?param_1 - floor ?param_2 - passenger)  
:precondition (and
		(lift-at ?param_1) (origin ?param_2 ?param_1)
)
:effect (and (boarded ?param_2) ))

(:action depart  
:parameters (?param_1 - floor ?param_2 - passenger)  
:precondition (and
		(boarded ?param_2) (destin ?param_2 ?param_1) (lift-at ?param_1)
)
:effect (and (served ?param_2) (not (boarded ?param_2))))

(:action up  
:parameters (?param_1 - floor ?param_2 - floor)  
:precondition (and
		(above ?param_1 ?param_2) (lift-at ?param_1)
)
:effect (and (lift-at ?param_2) (not (lift-at ?param_1))))

(:action down  
:parameters (?param_1 - floor ?param_2 - floor)  
:precondition (and
		(above ?param_2 ?param_1) (lift-at ?param_1)
)
:effect (and (lift-at ?param_2) (not (lift-at ?param_1))))
)


