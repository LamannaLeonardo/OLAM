
(define (domain parking)
 (:requirements :strips :typing )
 (:types car curb)
 (:predicates
    (at-curb ?car - car)
    (at-curb-num ?car - car ?curb - curb)
    (behind-car ?car ?front-car - car)
    (car-clear ?car - car)
    (curb-clear ?curb - curb)
 )




 (:action move-curb-to-curb  
:parameters (?param_1 - car ?param_2 ?param_3 - curb)  
:precondition (and
 (at-curb-num ?param_1 ?param_2) (curb-clear ?param_3) (car-clear ?param_1) (at-curb ?param_1)
)
:effect (and (at-curb-num ?param_1 ?param_3) (curb-clear ?param_2) (not (at-curb-num ?param_1 ?param_2)) (not (curb-clear ?param_3))))

 (:action move-curb-to-car  
:parameters (?param_1 - car ?param_2 - curb ?param_3 - car)  
:precondition (and
					(at-curb ?param_1) (at-curb ?param_3) (at-curb-num ?param_1 ?param_2) (car-clear ?param_1) (car-clear ?param_3)
)
:effect (and (curb-clear ?param_2) (behind-car ?param_1 ?param_3) (not (at-curb-num ?param_1 ?param_2)) (not (car-clear ?param_3)) (not (at-curb ?param_1))))

 (:action move-car-to-curb  
:parameters (?param_1 - car ?param_2 - car ?param_3 - curb)  
:precondition (and
					(at-curb ?param_2) (behind-car ?param_1 ?param_2) (car-clear ?param_1) (curb-clear ?param_3)
)
:effect (and (at-curb-num ?param_1 ?param_3) (at-curb ?param_1) (car-clear ?param_2) (not (behind-car ?param_1 ?param_2)) (not (curb-clear ?param_3))))

 (:action move-car-to-car  
:parameters (?param_1 - car ?param_2 - car ?param_3 - car)  
:precondition (and
					(at-curb ?param_2) (at-curb ?param_3) (behind-car ?param_1 ?param_2) (car-clear ?param_1) (car-clear ?param_3)
)
:effect (and (car-clear ?param_2) (behind-car ?param_1 ?param_3) (not (behind-car ?param_1 ?param_2)) (not (car-clear ?param_3))))
)











