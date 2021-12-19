(define (problem elevators-sequencedstrips-p4_4_1)
(:domain elevators-sequencedstrips)

(:objects 
n0 n1 n2 n3 n4  - count
p0 p1 p2 p3  - passenger
fast0  - fast-elevator
slow0-0 - slow-elevator
)

(:init
(next n0 n1) (next n1 n2) (next n2 n3) (next n3 n4) 

(above n0 n1) (above n0 n2) (above n0 n3) (above n0 n4) 
(above n1 n2) (above n1 n3) (above n1 n4) 
(above n2 n3) (above n2 n4) 
(above n3 n4) 

(lift-at fast0 n0)
(passengers fast0 n0)
(can-hold fast0 n1) 
(reachable-floor fast0 n0)(reachable-floor fast0 n2)(reachable-floor fast0 n4)

(lift-at slow0-0 n2)
(passengers slow0-0 n0)
(can-hold slow0-0 n1) 
(reachable-floor slow0-0 n0)(reachable-floor slow0-0 n1)(reachable-floor slow0-0 n2)(reachable-floor slow0-0 n3)(reachable-floor slow0-0 n4)

(passenger-at p0 n0)
(passenger-at p1 n1)
(passenger-at p2 n1)
(passenger-at p3 n1)

(= (travel-slow n0 n1) 6) (= (travel-slow n0 n2) 7) (= (travel-slow n0 n3) 8) (= (travel-slow n0 n4) 9) (= (travel-slow n1 n2) 6) (= (travel-slow n1 n3) 7) (= (travel-slow n1 n4) 8) (= (travel-slow n2 n3) 6) (= (travel-slow n2 n4) 7) (= (travel-slow n3 n4) 6) 


(= (travel-fast n0 n2) 7) (= (travel-fast n0 n4) 13) 

(= (travel-fast n2 n4) 7) 

(= (total-cost) 0)

)

(:goal
(and
(passenger-at p0 n3)
(passenger-at p1 n2)
(passenger-at p2 n0)
(passenger-at p3 n2)
))

(:metric minimize (total-cost))

)
