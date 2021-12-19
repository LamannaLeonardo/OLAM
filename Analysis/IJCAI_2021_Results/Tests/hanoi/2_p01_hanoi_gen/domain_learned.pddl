
(define (domain hanoi)
 (:requirements :strips :typing)
 (:types disc table - platform)
 (:predicates (clear ?x - platform)
    (on ?x - disc ?y - platform)
    (smaller ?x - platform ?y - disc))


 (:action move 
:parameters (?param_1 - disc ?param_2 - platform ?param_3 - platform) 
:precondition (and
 (on ?param_1 ?param_2) (smaller ?param_3 ?param_1) (clear ?param_1) (smaller ?param_2 ?param_1) (clear ?param_3)
)
:effect (and (clear ?param_2) (on ?param_1 ?param_3) (not (on ?param_1 ?param_2))))
)



