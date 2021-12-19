
(define (domain spanner)
(:requirements :typing :strips)
(:types
	location locatable - object
	man nut spanner - locatable
)

(:predicates
	(at ?m - locatable ?l - location)
	(carrying ?m - man ?s - spanner)
	(useable ?s - spanner)
	(link ?l1 - location ?l2 - location)
	(tightened ?n - nut)
	(loose ?n - nut))

(:action walk 
:parameters (?param_1 - location ?param_2 - location ?param_3 - man) 
:precondition (and
					(at ?param_3 ?param_1) (link ?param_1 ?param_2)
)
:effect (and (at ?param_3 ?param_2) (not (at ?param_3 ?param_1))))

(:action pickup_spanner 
:parameters (?param_1 - location ?param_2 - spanner ?param_3 - man) 
:precondition (and
					(at ?param_2 ?param_1) (at ?param_3 ?param_1) (useable ?param_2)
)
:effect (and (carrying ?param_3 ?param_2) (not (at ?param_2 ?param_1))))

(:action tighten_nut 
:parameters (?param_1 - location ?param_2 - spanner ?param_3 - man ?param_4 - nut) 
:precondition (and
					(at ?param_3 ?param_1) (at ?param_4 ?param_1) (carrying ?param_3 ?param_2) (loose ?param_4) (useable ?param_2)
)
:effect (and (tightened ?param_4) (not (useable ?param_2)) (not (loose ?param_4))))
)









