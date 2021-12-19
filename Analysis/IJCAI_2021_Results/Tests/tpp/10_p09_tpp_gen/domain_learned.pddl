
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
 (at ?param_1 ?param_2) (connected ?param_2 ?param_3) (connected ?param_3 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

(:action load 
:parameters (?param_1 - goods ?param_2 - truck ?param_3 - market ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
 (at ?param_2 ?param_3) (loaded ?param_1 ?param_2 ?param_6) (next ?param_5 ?param_4) (next ?param_7 ?param_6) (ready-to-load ?param_1 ?param_3 ?param_5)
)
:effect (and (loaded ?param_1 ?param_2 ?param_7) (ready-to-load ?param_1 ?param_3 ?param_4) (not (ready-to-load ?param_1 ?param_3 ?param_5)) (not (loaded ?param_1 ?param_2 ?param_6))))

(:action unload 
:parameters (?param_1 - goods ?param_2 - truck ?param_3 - depot ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
 (at ?param_2 ?param_3) (loaded ?param_1 ?param_2 ?param_5) (next ?param_5 ?param_4) (next ?param_7 ?param_6) (stored ?param_1 ?param_6)
)
:effect (and (stored ?param_1 ?param_7) (loaded ?param_1 ?param_2 ?param_4) (not (loaded ?param_1 ?param_2 ?param_5)) (not (stored ?param_1 ?param_6))))

(:action buy 
:parameters (?param_1 - truck ?param_2 - goods ?param_3 - market ?param_4 ?param_5 ?param_6 ?param_7 - level) 
:precondition (and
 (at ?param_1 ?param_3) (next ?param_5 ?param_4) (next ?param_7 ?param_6) (on-sale ?param_2 ?param_3 ?param_5) (ready-to-load ?param_2 ?param_3 ?param_6)
)
:effect (and (ready-to-load ?param_2 ?param_3 ?param_7) (on-sale ?param_2 ?param_3 ?param_4) (not (ready-to-load ?param_2 ?param_3 ?param_6)) (not (on-sale ?param_2 ?param_3 ?param_5))))
)

















