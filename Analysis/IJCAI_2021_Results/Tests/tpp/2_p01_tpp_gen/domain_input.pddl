
(define (domain tpp-propositional)
(:requirements :strips :typing)
(:types place locatable level - object
	depot market - place
	truck goods - locatable)

(:predicates (loaded ?g - goods ?t - truck ?l - level)
	     (ready-to-load ?g - goods ?m - market ?l - level)
	     (stored ?g - goods ?l - level)
	     (on-sale ?g - goods ?m -  market ?l - level)
	     (next ?l1 ?l2 - level)
	     (at ?t - truck ?p - place)
	     (connected ?p1 ?p2 - place))


(:action drive 
:parameters (?param_1 - truck ?param_2 ?param_3 - place) 
:precondition (and
		(at ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

(:action load 
:parameters (?param_1 - goods ?param_2 - truck ?param_3 - market ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
)
)
:effect (and  ))

(:action unload 
:parameters (?param_1 - goods ?param_2 - truck ?param_3 - depot ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
		(at ?param_2 ?param_3)
)
:effect (and  ))

(:action buy 
:parameters (?param_1 - truck ?param_2 - goods ?param_3 - market ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
		(at ?param_1 ?param_3)
)
:effect (and  ))
)













































