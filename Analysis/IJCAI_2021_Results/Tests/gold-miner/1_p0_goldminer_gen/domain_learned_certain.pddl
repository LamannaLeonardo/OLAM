

(define (domain gold-miner-typed)
(:requirements :typing)
(:types loc)

(:predicates 	
		(robot-at ?x - loc)
		(bomb-at ?x - loc )
		(laser-at ?x - loc)
		(soft-rock-at ?x - loc)
		(hard-rock-at ?x - loc)
		(gold-at ?x - loc)
		(arm-empty)
		(holds-bomb)
                (holds-laser)
		(holds-gold)
		(clear ?x - loc)		
		(connected ?x - loc ?y - loc)
)
 


(:action move 
:parameters (?param_1 - loc ?param_2 - loc) 
:precondition (and
		(clear ?param_2) (robot-at ?param_1)
)
:effect (and (robot-at ?param_2) (not (robot-at ?param_1))))

(:action pickup-laser 
:parameters (?param_1 - loc) 
:precondition (and
		(arm-empty) (laser-at ?param_1) (robot-at ?param_1)
)
:effect (and (holds-laser) (not (arm-empty)) (not (laser-at ?param_1))))

(:action pickup-bomb 
:parameters (?param_1 - loc) 
:precondition (and
		(arm-empty) (bomb-at ?param_1) (robot-at ?param_1)
)
:effect (and (holds-bomb) (not (arm-empty))))

(:action putdown-laser 
:parameters (?param_1 - loc) 
:precondition (and
		(holds-laser) (robot-at ?param_1)
)
:effect (and (laser-at ?param_1) (arm-empty) (not (holds-laser))))

(:action detonate-bomb 
:parameters (?param_1 - loc ?param_2 - loc) 
:precondition (and
		(holds-bomb) (robot-at ?param_1) (soft-rock-at ?param_2)
)
:effect (and (arm-empty) (clear ?param_2) (not (holds-bomb)) (not (soft-rock-at ?param_2))))

(:action fire-laser 
:parameters (?param_1 - loc ?param_2 - loc) 
:precondition (and
		(holds-laser) (robot-at ?param_1)
)
:effect (and (clear ?param_2) (not (hard-rock-at ?param_2))))

(:action pick-gold 
:parameters (?param_1 - loc) 
:precondition (and
		(arm-empty) (gold-at ?param_1) (robot-at ?param_1)
)
:effect (and (holds-gold) (not (arm-empty))))
)

























































