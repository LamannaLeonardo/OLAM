
(define (domain matching-bw-typed)
(:requirements :typing)
(:types block hand)
(:predicates (hand-positive ?h - hand)
             (hand-negative ?h - hand)
             (block-positive ?b - block)
             (block-negative ?b - block)
             (clear ?b - block)
             (on-table ?b - block)
             (empty ?h - hand)
             (holding ?h - hand ?b - block)
             (on ?b1 ?b2 - block)
	     (solid ?b - block))

(:action pickup 
:parameters (?param_1 - hand ?param_2 - block) 
:precondition (and
 (clear ?param_2) (empty ?param_1) (on-table ?param_2)
)
:effect (and (holding ?param_1 ?param_2) (not (clear ?param_2)) (not (empty ?param_1)) (not (on-table ?param_2))))

(:action putdown-pos-pos 
:parameters (?param_1 - hand ?param_2 - block) 
:precondition (and
 (block-positive ?param_2) (hand-positive ?param_1) (holding ?param_1 ?param_2)
)
:effect (and (on-table ?param_2) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2))))

(:action putdown-neg-neg 
:parameters (?param_1 - hand ?param_2 - block) 
:precondition (and
 (block-negative ?param_2) (hand-negative ?param_1) (holding ?param_1 ?param_2)
)
:effect (and (on-table ?param_2) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2))))

(:action putdown-pos-neg 
:parameters (?param_1 - hand ?param_2 - block) 
:precondition (and
 (block-negative ?param_2) (hand-positive ?param_1) (holding ?param_1 ?param_2)
)
:effect (and (on-table ?param_2) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (solid ?param_2))))

(:action putdown-neg-pos 
:parameters (?param_1 - hand ?param_2 - block) 
:precondition (and
 (block-positive ?param_2) (hand-negative ?param_1) (holding ?param_1 ?param_2)
)
:effect (and (on-table ?param_2) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (solid ?param_2))))

(:action stack-pos-pos 
:parameters (?param_1 - hand ?param_2 ?param_3 - block) 
:precondition (and
 (block-positive ?param_2) (clear ?param_3) (hand-positive ?param_1) (holding ?param_1 ?param_2) (solid ?param_3)
)
:effect (and (on ?param_2 ?param_3) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (clear ?param_3))))

(:action stack-neg-neg 
:parameters (?param_1 - hand ?param_2 ?param_3 - block) 
:precondition (and
 (block-negative ?param_2) (clear ?param_3) (hand-negative ?param_1) (holding ?param_1 ?param_2) (solid ?param_3)
)
:effect (and (on ?param_2 ?param_3) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (clear ?param_3))))

(:action stack-pos-neg 
:parameters (?param_1 - hand ?param_2 ?param_3 - block) 
:precondition (and
 (block-negative ?param_2) (clear ?param_3) (hand-positive ?param_1) (holding ?param_1 ?param_2) (solid ?param_3)
)
:effect (and (on ?param_2 ?param_3) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (solid ?param_2)) (not (clear ?param_3))))

(:action stack-neg-pos 
:parameters (?param_1 - hand ?param_2 ?param_3 - block) 
:precondition (and
 (block-positive ?param_2) (clear ?param_3) (hand-negative ?param_1) (holding ?param_1 ?param_2) (solid ?param_3)
)
:effect (and (on ?param_2 ?param_3) (clear ?param_2) (empty ?param_1) (not (holding ?param_1 ?param_2)) (not (solid ?param_2)) (not (clear ?param_3))))

(:action unstack 
:parameters (?param_1 - hand ?param_2 ?param_3 - block) 
:precondition (and
 (clear ?param_2) (empty ?param_1) (on ?param_2 ?param_3) (solid ?param_3)
)
:effect (and (holding ?param_1 ?param_2) (clear ?param_3) (not (clear ?param_2)) (not (on ?param_2 ?param_3)) (not (empty ?param_1))))
)

