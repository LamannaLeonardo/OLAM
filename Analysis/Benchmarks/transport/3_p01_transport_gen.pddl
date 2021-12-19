; Transport city-sequential-10nodes-400size-4degree-10mindistance-4trucks-8packages-248365seed

(define (problem transport-city-sequential-10nodes-400size-4degree-10mindistance-4trucks-8packages-248365seed)
 (:domain transport)
 (:objects
  city-loc-1 - location
  city-loc-2 - location
  city-loc-3 - location
  city-loc-4 - location
  city-loc-5 - location
  city-loc-6 - location
  city-loc-7 - location
  city-loc-8 - location
  city-loc-9 - location
  city-loc-10 - location
  truck-1 - vehicle
  truck-2 - vehicle
  truck-3 - vehicle
  truck-4 - vehicle
  package-1 - package
  package-2 - package
  package-3 - package
  package-4 - package
  package-5 - package
  package-6 - package
  package-7 - package
  package-8 - package
  capacity-0 - capacity-number
  capacity-1 - capacity-number
  capacity-2 - capacity-number
  capacity-3 - capacity-number
  capacity-4 - capacity-number
 )
 (:init
  (= (total-cost) 0)
  (capacity-predecessor capacity-0 capacity-1)
  (capacity-predecessor capacity-1 capacity-2)
  (capacity-predecessor capacity-2 capacity-3)
  (capacity-predecessor capacity-3 capacity-4)
  ; 119,79 -> 220,205
  (road city-loc-2 city-loc-1)
  (= (road-length city-loc-2 city-loc-1) 17)
  ; 220,205 -> 119,79
  (road city-loc-1 city-loc-2)
  (= (road-length city-loc-1 city-loc-2) 17)
  ; 219,138 -> 220,205
  (road city-loc-3 city-loc-1)
  (= (road-length city-loc-3 city-loc-1) 7)
  ; 220,205 -> 219,138
  (road city-loc-1 city-loc-3)
  (= (road-length city-loc-1 city-loc-3) 7)
  ; 219,138 -> 119,79
  (road city-loc-3 city-loc-2)
  (= (road-length city-loc-3 city-loc-2) 12)
  ; 119,79 -> 219,138
  (road city-loc-2 city-loc-3)
  (= (road-length city-loc-2 city-loc-3) 12)
  ; 35,136 -> 119,79
  (road city-loc-6 city-loc-2)
  (= (road-length city-loc-6 city-loc-2) 11)
  ; 119,79 -> 35,136
  (road city-loc-2 city-loc-6)
  (= (road-length city-loc-2 city-loc-6) 11)
  ; 42,109 -> 119,79
  (road city-loc-7 city-loc-2)
  (= (road-length city-loc-7 city-loc-2) 9)
  ; 119,79 -> 42,109
  (road city-loc-2 city-loc-7)
  (= (road-length city-loc-2 city-loc-7) 9)
  ; 42,109 -> 35,136
  (road city-loc-7 city-loc-6)
  (= (road-length city-loc-7 city-loc-6) 3)
  ; 35,136 -> 42,109
  (road city-loc-6 city-loc-7)
  (= (road-length city-loc-6 city-loc-7) 3)
  ; 262,247 -> 220,205
  (road city-loc-8 city-loc-1)
  (= (road-length city-loc-8 city-loc-1) 6)
  ; 220,205 -> 262,247
  (road city-loc-1 city-loc-8)
  (= (road-length city-loc-1 city-loc-8) 6)
  ; 262,247 -> 219,138
  (road city-loc-8 city-loc-3)
  (= (road-length city-loc-8 city-loc-3) 12)
  ; 219,138 -> 262,247
  (road city-loc-3 city-loc-8)
  (= (road-length city-loc-3 city-loc-8) 12)
  ; 262,247 -> 370,294
  (road city-loc-8 city-loc-5)
  (= (road-length city-loc-8 city-loc-5) 12)
  ; 370,294 -> 262,247
  (road city-loc-5 city-loc-8)
  (= (road-length city-loc-5 city-loc-8) 12)
  ; 223,275 -> 220,205
  (road city-loc-9 city-loc-1)
  (= (road-length city-loc-9 city-loc-1) 7)
  ; 220,205 -> 223,275
  (road city-loc-1 city-loc-9)
  (= (road-length city-loc-1 city-loc-9) 7)
  ; 223,275 -> 219,138
  (road city-loc-9 city-loc-3)
  (= (road-length city-loc-9 city-loc-3) 14)
  ; 219,138 -> 223,275
  (road city-loc-3 city-loc-9)
  (= (road-length city-loc-3 city-loc-9) 14)
  ; 223,275 -> 71,353
  (road city-loc-9 city-loc-4)
  (= (road-length city-loc-9 city-loc-4) 18)
  ; 71,353 -> 223,275
  (road city-loc-4 city-loc-9)
  (= (road-length city-loc-4 city-loc-9) 18)
  ; 223,275 -> 370,294
  (road city-loc-9 city-loc-5)
  (= (road-length city-loc-9 city-loc-5) 15)
  ; 370,294 -> 223,275
  (road city-loc-5 city-loc-9)
  (= (road-length city-loc-5 city-loc-9) 15)
  ; 223,275 -> 262,247
  (road city-loc-9 city-loc-8)
  (= (road-length city-loc-9 city-loc-8) 5)
  ; 262,247 -> 223,275
  (road city-loc-8 city-loc-9)
  (= (road-length city-loc-8 city-loc-9) 5)
  ; 261,390 -> 370,294
  (road city-loc-10 city-loc-5)
  (= (road-length city-loc-10 city-loc-5) 15)
  ; 370,294 -> 261,390
  (road city-loc-5 city-loc-10)
  (= (road-length city-loc-5 city-loc-10) 15)
  ; 261,390 -> 262,247
  (road city-loc-10 city-loc-8)
  (= (road-length city-loc-10 city-loc-8) 15)
  ; 262,247 -> 261,390
  (road city-loc-8 city-loc-10)
  (= (road-length city-loc-8 city-loc-10) 15)
  ; 261,390 -> 223,275
  (road city-loc-10 city-loc-9)
  (= (road-length city-loc-10 city-loc-9) 13)
  ; 223,275 -> 261,390
  (road city-loc-9 city-loc-10)
  (= (road-length city-loc-9 city-loc-10) 13)
  (at package-1 city-loc-6)
  (at package-2 city-loc-6)
  (at package-3 city-loc-8)
  (at package-4 city-loc-1)
  (at package-5 city-loc-3)
  (at package-6 city-loc-5)
  (at package-7 city-loc-6)
  (at package-8 city-loc-6)
  (at truck-1 city-loc-8)
  (capacity truck-1 capacity-3)
  (at truck-2 city-loc-8)
  (capacity truck-2 capacity-3)
  (at truck-3 city-loc-9)
  (capacity truck-3 capacity-2)
  (at truck-4 city-loc-2)
  (capacity truck-4 capacity-2)
 )
 (:goal (and
  (at package-1 city-loc-4)
  (at package-2 city-loc-5)
  (at package-3 city-loc-2)
  (at package-4 city-loc-9)
  (at package-5 city-loc-4)
  (at package-6 city-loc-4)
  (at package-7 city-loc-4)
  (at package-8 city-loc-7)
 ))
 (:metric minimize (total-cost))
)
