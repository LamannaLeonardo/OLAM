
(define (domain transport-strips)
(:requirements :typing )

(:types location fuellevel locatable - object
	package truck - locatable
)

(:predicates
(connected ?l1 ?l2 - location)
(at ?o - locatable ?l - location)
(in ?p - package ?t - truck)
(fuel ?t - truck ?level - fuellevel)
(fuelcost ?level - fuellevel ?l1 ?l2 - location)
(sum ?a ?b ?c - fuellevel)
)





(:action load
:parameters (?param_1 - package ?param_2 - truck ?param_3 - location)
:precondition (and
 (at ?param_1 ?param_3) (at ?param_2 ?param_3)
)
:effect (and (in ?param_1 ?param_2) (not (at ?param_1 ?param_3))))

(:action unload
:parameters (?param_1 - package ?param_2 - truck ?param_3 - location)
:precondition (and
 (at ?param_2 ?param_3) (in ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (in ?param_1 ?param_2))))

(:action drive
:parameters (?param_1 - truck ?param_2 - location ?param_3 - location ?param_4 - fuellevel ?param_5 - fuellevel ?param_6 - fuellevel)
:precondition (and
 (at ?param_1 ?param_2) (fuel ?param_1 ?param_6) (sum ?param_4 ?param_5 ?param_6) (fuelcost ?param_5 ?param_3 ?param_2) (connected ?param_2 ?param_3) (sum ?param_5 ?param_4 ?param_6) (fuelcost ?param_5 ?param_2 ?param_3) (connected ?param_3 ?param_2)
)
:effect (and (fuel ?param_1 ?param_4) (at ?param_1 ?param_3) (not (at ?param_1 ?param_2)) (not (fuel ?param_1 ?param_6))))
)




