
(define (domain elevators-sequencedstrips)
  (:requirements :typing)
  (:types 	elevator - object 
			slow-elevator fast-elevator - elevator
   			passenger - object
          	count - object
         )

(:predicates 
	(passenger-at ?person - passenger ?floor - count)
	(boarded ?person - passenger ?lift - elevator)
	(lift-at ?lift - elevator ?floor - count)
	(reachable-floor ?lift - elevator ?floor - count)
	(above ?floor1 - count ?floor2 - count)
	(passengers ?lift - elevator ?n - count)
	(can-hold ?lift - elevator ?n - count)
	(next ?n1 - count ?n2 - count)
)







(:action move-up-slow 
:parameters (?param_1 - slow-elevator ?param_2 - count ?param_3 - count ) 
:precondition (and
		(above ?param_2 ?param_3) (lift-at ?param_1 ?param_2)
)
:effect (and (lift-at ?param_1 ?param_3) (not (lift-at ?param_1 ?param_2))))

(:action move-down-slow 
:parameters (?param_1 - slow-elevator ?param_2 - count ?param_3 - count ) 
:precondition (and
		(above ?param_3 ?param_2) (lift-at ?param_1 ?param_2)
)
:effect (and (lift-at ?param_1 ?param_3) (not (lift-at ?param_1 ?param_2))))

(:action move-up-fast 
:parameters (?param_1 - fast-elevator ?param_2 - count ?param_3 - count ) 
:precondition (and
		(above ?param_2 ?param_3) (lift-at ?param_1 ?param_2) (reachable-floor ?param_1 ?param_3)
)
:effect (and (lift-at ?param_1 ?param_3) (not (lift-at ?param_1 ?param_2))))

(:action move-down-fast 
:parameters (?param_1 - fast-elevator ?param_2 - count ?param_3 - count ) 
:precondition (and
		(above ?param_3 ?param_2) (lift-at ?param_1 ?param_2) (reachable-floor ?param_1 ?param_3)
)
:effect (and (lift-at ?param_1 ?param_3) (not (lift-at ?param_1 ?param_2))))

(:action board 
:parameters (?param_1 - passenger ?param_2 - elevator ?param_3 - count ?param_4 - count ?param_5 - count) 
:precondition (and
		(can-hold ?param_2 ?param_5) (lift-at ?param_2 ?param_3) (next ?param_4 ?param_5) (passenger-at ?param_1 ?param_3) (passengers ?param_2 ?param_4)
)
:effect (and (passengers ?param_2 ?param_5) (boarded ?param_1 ?param_2) (not (passenger-at ?param_1 ?param_3)) (not (passengers ?param_2 ?param_4))))

(:action leave 
:parameters (?param_1 - passenger ?param_2 - elevator ?param_3 - count ?param_4 - count ?param_5 - count) 
:precondition (and
		(boarded ?param_1 ?param_2) (lift-at ?param_2 ?param_3) (next ?param_5 ?param_4) (passengers ?param_2 ?param_4)
)
:effect (and (passengers ?param_2 ?param_5) (passenger-at ?param_1 ?param_3) (not (boarded ?param_1 ?param_2)) (not (passengers ?param_2 ?param_4))))
)







































































