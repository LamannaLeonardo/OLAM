
(define (domain gripper-strips)
 (:requirements :strips :typing)
 (:types room object robot gripper)
 (:predicates (at-robby ?r - robot ?x - room)
 	      (at ?o - object ?x - room)
	      (free ?r - robot ?g - gripper)
	      (carry ?r - robot ?o - object ?g - gripper))


 (:action move 
:parameters (?param_1 - robot ?param_2 ?param_3 - room) 
:precondition (and
 (at-robby ?param_1 ?param_2)
)
:effect (and (at-robby ?param_1 ?param_3) (not (at-robby ?param_1 ?param_2))))

 (:action pick 
:parameters (?param_1 - robot ?param_2 - object ?param_3 - room ?param_4 - gripper) 
:precondition (and
 (at ?param_2 ?param_3) (at-robby ?param_1 ?param_3) (free ?param_1 ?param_4)
)
:effect (and (carry ?param_1 ?param_2 ?param_4) (not (free ?param_1 ?param_4)) (not (at ?param_2 ?param_3))))

 (:action drop 
:parameters (?param_1 - robot ?param_2 - object ?param_3 - room ?param_4 - gripper) 
:precondition (and
 (at-robby ?param_1 ?param_3) (carry ?param_1 ?param_2 ?param_4)
)
:effect (and (free ?param_1 ?param_4) (at ?param_2 ?param_3) (not (carry ?param_1 ?param_2 ?param_4))))
)



