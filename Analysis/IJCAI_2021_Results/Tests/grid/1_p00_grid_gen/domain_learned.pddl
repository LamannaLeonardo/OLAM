
(define (domain grid)
(:requirements :typing)
(:types place key shape)

(:predicates
		(conn ?x - place ?y - place)
		(key-shape ?k - key ?s - shape)
		(lock-shape ?x - place ?s - shape)
		(at ?r - key ?x - place)
		(at-robot ?x - place)
		(locked ?x - place)
		(holding ?k - key)
		(open ?x - place)
		(arm-empty))


(:action unlock  
:parameters (?param_1 - place ?param_2 - place ?param_3 - key ?param_4 - shape)  
:precondition (and
					(at-robot ?param_1) (conn ?param_1 ?param_2) (conn ?param_2 ?param_1) (holding ?param_3) (key-shape ?param_3 ?param_4) (lock-shape ?param_2 ?param_4) (locked ?param_2) (open ?param_1)
)
:effect (and (open ?param_2) (not (locked ?param_2))))

(:action move  
:parameters (?param_1 - place ?param_2 - place)  
:precondition (and
					(at-robot ?param_1) (conn ?param_1 ?param_2) (conn ?param_2 ?param_1) (open ?param_1) (open ?param_2)
)
:effect (and (at-robot ?param_2) (not (at-robot ?param_1))))

(:action pickup  
:parameters (?param_1 - place ?param_2 - key)  
:precondition (and
					(arm-empty) (at ?param_2 ?param_1) (at-robot ?param_1) (open ?param_1)
)
:effect (and (holding ?param_2) (not (at ?param_2 ?param_1)) (not (arm-empty))))

(:action pickup-and-loose  
:parameters (?param_1 - place ?param_2 - key ?param_3 - key)  
:precondition (and
)
:effect (and ))

(:action putdown  
:parameters (?param_1 - place ?param_2 - key)  
:precondition (and
					(at-robot ?param_1) (holding ?param_2) (open ?param_1)
)
:effect (and (at ?param_2 ?param_1) (arm-empty) (not (holding ?param_2))))
)























