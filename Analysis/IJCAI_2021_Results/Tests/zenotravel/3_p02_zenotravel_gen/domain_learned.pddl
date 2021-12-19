
(define (domain zeno-travel)
(:requirements :typing)
(:types aircraft person - either_aircraft_person
either_aircraft_person city flevel - object)
(:predicates (at ?x - either_aircraft_person ?c - city)
             (in ?p - person ?a - aircraft)
	     (fuel-level ?a - aircraft ?l - flevel)
	     (next ?l1 ?l2 - flevel))



(:action board 
:parameters (?param_1 - person ?param_2 - aircraft ?param_3 - city) 
:precondition (and
 (at ?param_1 ?param_3) (at ?param_2 ?param_3)
)
:effect (and (in ?param_1 ?param_2) (not (at ?param_1 ?param_3))))

(:action debark 
:parameters (?param_1 - person ?param_2 - aircraft ?param_3 - city) 
:precondition (and
 (at ?param_2 ?param_3) (in ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (in ?param_1 ?param_2))))

(:action fly 
:parameters (?param_1 - aircraft ?param_2 ?param_3 - city ?param_4 ?param_5 - flevel) 
:precondition (and
 (at ?param_1 ?param_2) (fuel-level ?param_1 ?param_4) (next ?param_5 ?param_4)
)
:effect (and (fuel-level ?param_1 ?param_5) (at ?param_1 ?param_3) (not (at ?param_1 ?param_2)) (not (fuel-level ?param_1 ?param_4))))

(:action zoom 
:parameters (?param_1 - aircraft ?param_2 ?param_3 - city ?param_4 ?param_5 ?param_6 - flevel) 
:precondition (and
 (at ?param_1 ?param_2) (fuel-level ?param_1 ?param_4) (next ?param_5 ?param_4) (next ?param_6 ?param_5)
)
:effect (and (fuel-level ?param_1 ?param_6) (at ?param_1 ?param_3) (not (fuel-level ?param_1 ?param_4)) (not (at ?param_1 ?param_2))))

(:action refuel 
:parameters (?param_1 - aircraft ?param_2 - city ?param_3 - flevel ?param_4 - flevel) 
:precondition (and
 (at ?param_1 ?param_2) (fuel-level ?param_1 ?param_3) (next ?param_3 ?param_4)
)
:effect (and (fuel-level ?param_1 ?param_4) (not (fuel-level ?param_1 ?param_3))))
)

