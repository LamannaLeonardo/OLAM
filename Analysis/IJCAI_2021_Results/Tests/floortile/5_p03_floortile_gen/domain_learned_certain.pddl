

(define (domain floor-tile)
(:requirements :typing)
(:types robot tile color - object)

(:predicates 	
		(robot-at ?r - robot ?x - tile)
		(up ?x - tile ?y - tile)
		(down ?x - tile ?y - tile)
		(right ?x - tile ?y - tile)
		(left ?x - tile ?y - tile)
		
		(clear ?x - tile)
                (painted ?x - tile ?c - color)
		(robot-has ?r - robot ?c - color)
                (available-color ?c - color)
                (free-color ?r - robot))




(:action change-color 
:parameters (?param_1 - robot ?param_2 - color ?param_3 - color) 
:precondition (and
		(robot-has ?param_1 ?param_2)
)
:effect (and (robot-has ?param_1 ?param_3) (not (robot-has ?param_1 ?param_2))))

(:action paint-up 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile ?param_4 - color) 
:precondition (and
		(clear ?param_2) (robot-at ?param_1 ?param_3) (robot-has ?param_1 ?param_4)
)
:effect (and (painted ?param_2 ?param_4) (not (clear ?param_2))))

(:action paint-down 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile ?param_4 - color) 
:precondition (and
		(clear ?param_2) (robot-at ?param_1 ?param_3) (robot-has ?param_1 ?param_4)
)
:effect (and (painted ?param_2 ?param_4) (not (clear ?param_2))))

(:action up 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile) 
:precondition (and
		(clear ?param_3) (robot-at ?param_1 ?param_2)
)
:effect (and (robot-at ?param_1 ?param_3) (clear ?param_2) (not (robot-at ?param_1 ?param_2)) (not (clear ?param_3))))

(:action down 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile) 
:precondition (and
		(clear ?param_3) (robot-at ?param_1 ?param_2)
)
:effect (and (robot-at ?param_1 ?param_3) (clear ?param_2) (not (robot-at ?param_1 ?param_2)) (not (clear ?param_3))))

(:action right 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile) 
:precondition (and
		(clear ?param_3) (robot-at ?param_1 ?param_2)
)
:effect (and (robot-at ?param_1 ?param_3) (clear ?param_2) (not (robot-at ?param_1 ?param_2)) (not (clear ?param_3))))

(:action left 
:parameters (?param_1 - robot ?param_2 - tile ?param_3 - tile) 
:precondition (and
		(clear ?param_3) (robot-at ?param_1 ?param_2)
)
:effect (and (robot-at ?param_1 ?param_3) (clear ?param_2) (not (robot-at ?param_1 ?param_2)) (not (clear ?param_3))))
)


