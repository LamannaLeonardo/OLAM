
(define (domain driverlog)
  (:requirements :typing)
  (:types
        location locatable - object
		driver truck obj - locatable

  )
  (:predicates
		(at ?obj - locatable ?loc - location)
		(in ?obj1 - obj ?obj - truck)
		(driving ?d - driver ?v - truck)
		(link ?x ?y - location) (path ?x ?y - location)
		(empty ?v - truck))



(:action load-truck 
:parameters (?param_1 - obj ?param_2 - truck ?param_3 - location) 
:precondition (and
 (at ?param_1 ?param_3) (at ?param_2 ?param_3)
)
:effect (and (in ?param_1 ?param_2) (not (at ?param_1 ?param_3))))

(:action unload-truck 
:parameters (?param_1 - obj ?param_2 - truck ?param_3 - location) 
:precondition (and
 (in ?param_1 ?param_2) (at ?param_2 ?param_3)
)
:effect (and (at ?param_1 ?param_3) (not (in ?param_1 ?param_2))))

(:action board-truck 
:parameters (?param_1 - driver ?param_2 - truck ?param_3 - location) 
:precondition (and
 (at ?param_1 ?param_3) (empty ?param_2) (at ?param_2 ?param_3)
)
:effect (and (driving ?param_1 ?param_2) (not (at ?param_1 ?param_3)) (not (empty ?param_2))))

(:action disembark-truck 
:parameters (?param_1 - driver ?param_2 - truck ?param_3 - location) 
:precondition (and
 (driving ?param_1 ?param_2) (at ?param_2 ?param_3)
)
:effect (and (empty ?param_2) (at ?param_1 ?param_3) (not (driving ?param_1 ?param_2))))

(:action drive-truck 
:parameters (?param_1 - truck ?param_2 - location ?param_3 - location ?param_4 - driver) 
:precondition (and
					(at ?param_1 ?param_2) (driving ?param_4 ?param_1) (link ?param_2 ?param_3) (link ?param_3 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

(:action walk 
:parameters (?param_1 - driver ?param_2 - location ?param_3 - location) 
:precondition (and
					(at ?param_1 ?param_2) (path ?param_2 ?param_3) (path ?param_3 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))
)











