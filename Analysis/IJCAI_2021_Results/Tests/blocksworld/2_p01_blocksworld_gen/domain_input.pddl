
(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block)
	       (ontable ?x - block)
	       (clear ?x - block)
	       (handempty)
	       (holding ?x - block)
	       )


 (:action pick-up  
:parameters (?param_1 - block)  
:precondition (and
		(clear ?param_1) (handempty) (ontable ?param_1)
)
:effect (and (holding ?param_1) (not (clear ?param_1)) (not (handempty)) (not (ontable ?param_1))))

 (:action put-down  
:parameters (?param_1 - block)  
:precondition (and
		(holding ?param_1)
)
:effect (and (handempty) (ontable ?param_1) (clear ?param_1) (not (holding ?param_1))))

 (:action stack  
:parameters (?param_1 - block ?param_2 - block)  
:precondition (and
		(clear ?param_2) (holding ?param_1)
)
:effect (and (handempty) (on ?param_1 ?param_2) (clear ?param_1) (not (clear ?param_2)) (not (holding ?param_1))))

 (:action unstack  
:parameters (?param_1 - block ?param_2 - block)  
:precondition (and
		(clear ?param_1) (handempty) (on ?param_1 ?param_2)
)
:effect (and (clear ?param_2) (holding ?param_1) (not (clear ?param_1)) (not (on ?param_1 ?param_2)) (not (handempty))))
)
















