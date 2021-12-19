

(define (domain transport)
  (:requirements :typing)
  (:types
        location target locatable - object
        vehicle package - locatable
        capacity-number - object
  )

  (:predicates 
     (road ?l1 ?l2 - location)
     (at ?x - locatable ?v - location)
     (in ?x - package ?v - vehicle)
     (capacity ?v - vehicle ?s1 - capacity-number)
     (capacity-predecessor ?s1 ?s2 - capacity-number)
  )







 (:action drive 
:parameters (?param_1 - vehicle ?param_2 ?param_3 - location) 
:precondition (and
		(at ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

 (:action pick-up 
:parameters (?param_1 - vehicle ?param_2 - location ?param_3 - package ?param_4 ?param_5 - capacity-number) 
:precondition (and
		(at ?param_1 ?param_2) (at ?param_3 ?param_2) (capacity ?param_1 ?param_5) (capacity-predecessor ?param_4 ?param_5)
)
:effect (and (in ?param_3 ?param_1) (capacity ?param_1 ?param_4) (not (at ?param_3 ?param_2)) (not (capacity ?param_1 ?param_5))))

 (:action drop 
:parameters (?param_1 - vehicle ?param_2 - location ?param_3 - package ?param_4 ?param_5 - capacity-number) 
:precondition (and
		(at ?param_1 ?param_2) (capacity ?param_1 ?param_4) (capacity-predecessor ?param_4 ?param_5) (in ?param_3 ?param_1)
)
:effect (and (capacity ?param_1 ?param_5) (at ?param_3 ?param_2) (not (in ?param_3 ?param_1)) (not (capacity ?param_1 ?param_4))))
)








