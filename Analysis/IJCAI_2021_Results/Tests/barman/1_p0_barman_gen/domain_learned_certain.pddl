
(define (domain barman)
  (:requirements :strips :typing)
  (:types hand level beverage dispenser container - object
  	  ingredient cocktail - beverage
          shot shaker - container)
  (:predicates  (ontable ?c - container)
                (holding ?h - hand ?c - container)
		(handempty ?h - hand)
		(empty ?c - container)
                (contains ?c - container ?b - beverage)
		(clean ?c - container)
                (used ?c - container ?b - beverage)
                (dispenses ?d - dispenser ?i - ingredient)
		(shaker-empty-level ?s - shaker ?l - level)
		(shaker-level ?s - shaker ?l - level)
		(next ?l1 ?l2 - level)
		(unshaked ?s - shaker)
		(shaked ?s - shaker)
                (cocktail-part1 ?c - cocktail ?i - ingredient)
                (cocktail-part2 ?c - cocktail ?i - ingredient))
		

 (:action grasp 
:parameters (?param_1 - hand ?param_2 - container) 
:precondition (and
		(handempty ?param_1) (ontable ?param_2)
)
:effect (and (holding ?param_1 ?param_2) (not (ontable ?param_2)) (not (handempty ?param_1))))

 (:action leave 
:parameters (?param_1 - hand ?param_2 - container) 
:precondition (and
		(holding ?param_1 ?param_2)
)
:effect (and (ontable ?param_2) (handempty ?param_1) (not (holding ?param_1 ?param_2))))

 (:action fill-shot 
:parameters (?param_1 - shot ?param_2 - ingredient ?param_3 ?param_4 - hand ?param_5 - dispenser) 
:precondition (and
		(clean ?param_1) (dispenses ?param_5 ?param_2) (handempty ?param_4) (holding ?param_3 ?param_1)
)
:effect (and (used ?param_1 ?param_2) (contains ?param_1 ?param_2) (not (empty ?param_1)) (not (clean ?param_1))))

 (:action refill-shot 
:parameters (?param_1 - shot ?param_2 - ingredient ?param_3 ?param_4 - hand ?param_5 - dispenser) 
:precondition (and
		(dispenses ?param_5 ?param_2) (empty ?param_1) (handempty ?param_4) (holding ?param_3 ?param_1) (used ?param_1 ?param_2)
)
:effect (and (contains ?param_1 ?param_2) (not (empty ?param_1))))

 (:action empty-shot 
:parameters (?param_1 - hand ?param_2 - shot ?param_3 - beverage) 
:precondition (and
		(contains ?param_2 ?param_3) (holding ?param_1 ?param_2)
)
:effect (and (empty ?param_2) (not (contains ?param_2 ?param_3))))

 (:action clean-shot   
:parameters (?param_1 - shot ?param_2 - beverage ?param_3 ?param_4 - hand) 
:precondition (and
		(empty ?param_1) (handempty ?param_4) (holding ?param_3 ?param_1) (used ?param_1 ?param_2)
)
:effect (and (clean ?param_1) (not (used ?param_1 ?param_2))))

 (:action pour-shot-to-clean-shaker 
:parameters (?param_1 - shot ?param_2 - ingredient ?param_3 - shaker ?param_4 - hand ?param_5 ?param_6 - level) 
:precondition (and
		(contains ?param_1 ?param_2) (holding ?param_4 ?param_1) (next ?param_5 ?param_6)
)
:effect (and (unshaked ?param_3) (shaker-level ?param_3 ?param_6) (empty ?param_1) (contains ?param_3 ?param_2) (not (empty ?param_3)) (not (contains ?param_1 ?param_2)) (not (clean ?param_3)) (not (shaker-level ?param_3 ?param_5))))

 (:action pour-shot-to-used-shaker 
:parameters (?param_1 - shot ?param_2 - ingredient ?param_3 - shaker ?param_4 - hand ?param_5 ?param_6 - level) 
:precondition (and
		(contains ?param_1 ?param_2) (next ?param_5 ?param_6) (shaker-level ?param_3 ?param_5)
)
:effect (and (shaker-level ?param_3 ?param_6) (empty ?param_1) (not (contains ?param_1 ?param_2)) (not (shaker-level ?param_3 ?param_5))))

 (:action empty-shaker 
:parameters (?param_1 - hand ?param_2 - shaker ?param_3 - cocktail ?param_4 ?param_5 - level) 
:precondition (and
)
:effect (and ))

 (:action clean-shaker   
:parameters (?param_1 ?param_2 - hand ?param_3 - shaker) 
:precondition (and
		(handempty ?param_2) (holding ?param_1 ?param_3)
)
:effect (and  ))

 (:action shake   
:parameters (?param_1 - cocktail ?param_2 ?param_3 - ingredient ?param_4 - shaker ?param_5 ?param_6 - hand) 
:precondition (and
)
:effect (and ))

 (:action pour-shaker-to-shot 
:parameters (?param_1 - beverage ?param_2 - shot ?param_3 - hand ?param_4 - shaker ?param_5 ?param_6 - level) 
:precondition (and
)
:effect (and ))
)


























































































































