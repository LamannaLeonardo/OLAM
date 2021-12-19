
(define (domain n-puzzle-typed)
  (:requirements :typing)
  (:types position tile)
  (:predicates (at ?tile - tile ?position - position)
	       (neighbor ?p1 - position ?p2 - position) 
	       (empty ?position - position)
   )


 (:action move 
:parameters (?param_1 - tile ?param_2 ?param_3 - position) 
:precondition (and
 (at ?param_1 ?param_2) (empty ?param_3) (neighbor ?param_3 ?param_2) (neighbor ?param_2 ?param_3)
)
:effect (and (empty ?param_2) (at ?param_1 ?param_3) (not (empty ?param_3)) (not (at ?param_1 ?param_2))))
)

