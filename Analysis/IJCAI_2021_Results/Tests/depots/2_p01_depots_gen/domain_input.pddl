
(define (domain depots)
(:requirements :strips :typing)
(:types place locatable - object
	depot distributor - place
        truck hoist surface - locatable
        pallet crate - surface)

(:predicates (at ?x - locatable ?y - place)
             (on ?x - crate ?y - surface)
             (in ?x - crate ?y - truck)
             (lifting ?x - hoist ?y - crate)
             (available ?x - hoist)
             (clear ?x - surface))


(:action drive 
:parameters (?param_1 - truck ?param_2 - place ?param_3 - place) 
:precondition (and
		(at ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

(:action lift 
:parameters (?param_1 - hoist ?param_2 - crate ?param_3 - surface ?param_4 - place) 
:precondition (and
		(at ?param_1 ?param_4) (available ?param_1) (clear ?param_2) (on ?param_2 ?param_3)
)
:effect (and (lifting ?param_1 ?param_2) (clear ?param_3) (not (available ?param_1)) (not (clear ?param_2)) (not (at ?param_2 ?param_4)) (not (on ?param_2 ?param_3))))

(:action drop 
:parameters (?param_1 - hoist ?param_2 - crate ?param_3 - surface ?param_4 - place) 
:precondition (and
		(at ?param_1 ?param_4) (at ?param_3 ?param_4) (clear ?param_3) (lifting ?param_1 ?param_2)
)
:effect (and (clear ?param_2) (on ?param_2 ?param_3) (at ?param_2 ?param_4) (available ?param_1) (not (lifting ?param_1 ?param_2)) (not (clear ?param_3))))

(:action load 
:parameters (?param_1 - hoist ?param_2 - crate ?param_3 - truck ?param_4 - place) 
:precondition (and
		(at ?param_1 ?param_4) (at ?param_3 ?param_4) (lifting ?param_1 ?param_2)
)
:effect (and (in ?param_2 ?param_3) (available ?param_1) (not (lifting ?param_1 ?param_2))))

(:action unload 
:parameters (?param_1 - hoist ?param_2 - crate ?param_3 - truck ?param_4 - place) 
:precondition (and
		(at ?param_1 ?param_4) (at ?param_3 ?param_4) (available ?param_1) (in ?param_2 ?param_3)
)
:effect (and (lifting ?param_1 ?param_2) (not (available ?param_1)) (not (in ?param_2 ?param_3))))
)

















