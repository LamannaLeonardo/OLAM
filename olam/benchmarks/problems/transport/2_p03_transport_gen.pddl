; Transport two-cities-sequential-8nodes-400size-4degree-10mindistance-3trucks-6packages-3657131seed

(define (problem transport-two-cities-sequential-8nodes-400size-4degree-10mindistance-3trucks-6packages-3657131seed)
 (:domain transport)
 (:objects
  city-1-loc-1 - location
  city-2-loc-1 - location
  city-1-loc-2 - location
  city-2-loc-2 - location
  city-1-loc-3 - location
  city-2-loc-3 - location
  city-1-loc-4 - location
  city-2-loc-4 - location
  city-1-loc-5 - location
  city-2-loc-5 - location
  city-1-loc-6 - location
  city-2-loc-6 - location
  city-1-loc-7 - location
  city-2-loc-7 - location
  city-1-loc-8 - location
  city-2-loc-8 - location
  truck-1 - vehicle
  truck-2 - vehicle
  truck-3 - vehicle
  package-1 - package
  package-2 - package
  package-3 - package
  package-4 - package
  package-5 - package
  package-6 - package
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
  ; 199,256 -> 363,252
  (road city-1-loc-3 city-1-loc-1)
  (= (road-length city-1-loc-3 city-1-loc-1) 17)
  ; 363,252 -> 199,256
  (road city-1-loc-1 city-1-loc-3)
  (= (road-length city-1-loc-1 city-1-loc-3) 17)
  ; 199,256 -> 103,109
  (road city-1-loc-3 city-1-loc-2)
  (= (road-length city-1-loc-3 city-1-loc-2) 18)
  ; 103,109 -> 199,256
  (road city-1-loc-2 city-1-loc-3)
  (= (road-length city-1-loc-2 city-1-loc-3) 18)
  ; 182,128 -> 103,109
  (road city-1-loc-5 city-1-loc-2)
  (= (road-length city-1-loc-5 city-1-loc-2) 9)
  ; 103,109 -> 182,128
  (road city-1-loc-2 city-1-loc-5)
  (= (road-length city-1-loc-2 city-1-loc-5) 9)
  ; 182,128 -> 199,256
  (road city-1-loc-5 city-1-loc-3)
  (= (road-length city-1-loc-5 city-1-loc-3) 13)
  ; 199,256 -> 182,128
  (road city-1-loc-3 city-1-loc-5)
  (= (road-length city-1-loc-3 city-1-loc-5) 13)
  ; 182,128 -> 338,61
  (road city-1-loc-5 city-1-loc-4)
  (= (road-length city-1-loc-5 city-1-loc-4) 17)
  ; 338,61 -> 182,128
  (road city-1-loc-4 city-1-loc-5)
  (= (road-length city-1-loc-4 city-1-loc-5) 17)
  ; 397,193 -> 363,252
  (road city-1-loc-6 city-1-loc-1)
  (= (road-length city-1-loc-6 city-1-loc-1) 7)
  ; 363,252 -> 397,193
  (road city-1-loc-1 city-1-loc-6)
  (= (road-length city-1-loc-1 city-1-loc-6) 7)
  ; 397,193 -> 338,61
  (road city-1-loc-6 city-1-loc-4)
  (= (road-length city-1-loc-6 city-1-loc-4) 15)
  ; 338,61 -> 397,193
  (road city-1-loc-4 city-1-loc-6)
  (= (road-length city-1-loc-4 city-1-loc-6) 15)
  ; 226,339 -> 363,252
  (road city-1-loc-7 city-1-loc-1)
  (= (road-length city-1-loc-7 city-1-loc-1) 17)
  ; 363,252 -> 226,339
  (road city-1-loc-1 city-1-loc-7)
  (= (road-length city-1-loc-1 city-1-loc-7) 17)
  ; 226,339 -> 199,256
  (road city-1-loc-7 city-1-loc-3)
  (= (road-length city-1-loc-7 city-1-loc-3) 9)
  ; 199,256 -> 226,339
  (road city-1-loc-3 city-1-loc-7)
  (= (road-length city-1-loc-3 city-1-loc-7) 9)
  ; 363,0 -> 338,61
  (road city-1-loc-8 city-1-loc-4)
  (= (road-length city-1-loc-8 city-1-loc-4) 7)
  ; 338,61 -> 363,0
  (road city-1-loc-4 city-1-loc-8)
  (= (road-length city-1-loc-4 city-1-loc-8) 7)
  ; 1014,20 -> 1105,30
  (road city-2-loc-2 city-2-loc-1)
  (= (road-length city-2-loc-2 city-2-loc-1) 10)
  ; 1105,30 -> 1014,20
  (road city-2-loc-1 city-2-loc-2)
  (= (road-length city-2-loc-1 city-2-loc-2) 10)
  ; 906,1 -> 1014,20
  (road city-2-loc-3 city-2-loc-2)
  (= (road-length city-2-loc-3 city-2-loc-2) 11)
  ; 1014,20 -> 906,1
  (road city-2-loc-2 city-2-loc-3)
  (= (road-length city-2-loc-2 city-2-loc-3) 11)
  ; 1160,122 -> 1105,30
  (road city-2-loc-5 city-2-loc-1)
  (= (road-length city-2-loc-5 city-2-loc-1) 11)
  ; 1105,30 -> 1160,122
  (road city-2-loc-1 city-2-loc-5)
  (= (road-length city-2-loc-1 city-2-loc-5) 11)
  ; 1160,122 -> 1014,20
  (road city-2-loc-5 city-2-loc-2)
  (= (road-length city-2-loc-5 city-2-loc-2) 18)
  ; 1014,20 -> 1160,122
  (road city-2-loc-2 city-2-loc-5)
  (= (road-length city-2-loc-2 city-2-loc-5) 18)
  ; 1160,122 -> 1074,251
  (road city-2-loc-5 city-2-loc-4)
  (= (road-length city-2-loc-5 city-2-loc-4) 16)
  ; 1074,251 -> 1160,122
  (road city-2-loc-4 city-2-loc-5)
  (= (road-length city-2-loc-4 city-2-loc-5) 16)
  ; 1046,166 -> 1105,30
  (road city-2-loc-6 city-2-loc-1)
  (= (road-length city-2-loc-6 city-2-loc-1) 15)
  ; 1105,30 -> 1046,166
  (road city-2-loc-1 city-2-loc-6)
  (= (road-length city-2-loc-1 city-2-loc-6) 15)
  ; 1046,166 -> 1014,20
  (road city-2-loc-6 city-2-loc-2)
  (= (road-length city-2-loc-6 city-2-loc-2) 15)
  ; 1014,20 -> 1046,166
  (road city-2-loc-2 city-2-loc-6)
  (= (road-length city-2-loc-2 city-2-loc-6) 15)
  ; 1046,166 -> 1074,251
  (road city-2-loc-6 city-2-loc-4)
  (= (road-length city-2-loc-6 city-2-loc-4) 9)
  ; 1074,251 -> 1046,166
  (road city-2-loc-4 city-2-loc-6)
  (= (road-length city-2-loc-4 city-2-loc-6) 9)
  ; 1046,166 -> 1160,122
  (road city-2-loc-6 city-2-loc-5)
  (= (road-length city-2-loc-6 city-2-loc-5) 13)
  ; 1160,122 -> 1046,166
  (road city-2-loc-5 city-2-loc-6)
  (= (road-length city-2-loc-5 city-2-loc-6) 13)
  ; 1085,70 -> 1105,30
  (road city-2-loc-7 city-2-loc-1)
  (= (road-length city-2-loc-7 city-2-loc-1) 5)
  ; 1105,30 -> 1085,70
  (road city-2-loc-1 city-2-loc-7)
  (= (road-length city-2-loc-1 city-2-loc-7) 5)
  ; 1085,70 -> 1014,20
  (road city-2-loc-7 city-2-loc-2)
  (= (road-length city-2-loc-7 city-2-loc-2) 9)
  ; 1014,20 -> 1085,70
  (road city-2-loc-2 city-2-loc-7)
  (= (road-length city-2-loc-2 city-2-loc-7) 9)
  ; 1085,70 -> 1074,251
  (road city-2-loc-7 city-2-loc-4)
  (= (road-length city-2-loc-7 city-2-loc-4) 19)
  ; 1074,251 -> 1085,70
  (road city-2-loc-4 city-2-loc-7)
  (= (road-length city-2-loc-4 city-2-loc-7) 19)
  ; 1085,70 -> 1160,122
  (road city-2-loc-7 city-2-loc-5)
  (= (road-length city-2-loc-7 city-2-loc-5) 10)
  ; 1160,122 -> 1085,70
  (road city-2-loc-5 city-2-loc-7)
  (= (road-length city-2-loc-5 city-2-loc-7) 10)
  ; 1085,70 -> 1046,166
  (road city-2-loc-7 city-2-loc-6)
  (= (road-length city-2-loc-7 city-2-loc-6) 11)
  ; 1046,166 -> 1085,70
  (road city-2-loc-6 city-2-loc-7)
  (= (road-length city-2-loc-6 city-2-loc-7) 11)
  ; 958,52 -> 1105,30
  (road city-2-loc-8 city-2-loc-1)
  (= (road-length city-2-loc-8 city-2-loc-1) 15)
  ; 1105,30 -> 958,52
  (road city-2-loc-1 city-2-loc-8)
  (= (road-length city-2-loc-1 city-2-loc-8) 15)
  ; 958,52 -> 1014,20
  (road city-2-loc-8 city-2-loc-2)
  (= (road-length city-2-loc-8 city-2-loc-2) 7)
  ; 1014,20 -> 958,52
  (road city-2-loc-2 city-2-loc-8)
  (= (road-length city-2-loc-2 city-2-loc-8) 7)
  ; 958,52 -> 906,1
  (road city-2-loc-8 city-2-loc-3)
  (= (road-length city-2-loc-8 city-2-loc-3) 8)
  ; 906,1 -> 958,52
  (road city-2-loc-3 city-2-loc-8)
  (= (road-length city-2-loc-3 city-2-loc-8) 8)
  ; 958,52 -> 1046,166
  (road city-2-loc-8 city-2-loc-6)
  (= (road-length city-2-loc-8 city-2-loc-6) 15)
  ; 1046,166 -> 958,52
  (road city-2-loc-6 city-2-loc-8)
  (= (road-length city-2-loc-6 city-2-loc-8) 15)
  ; 958,52 -> 1085,70
  (road city-2-loc-8 city-2-loc-7)
  (= (road-length city-2-loc-8 city-2-loc-7) 13)
  ; 1085,70 -> 958,52
  (road city-2-loc-7 city-2-loc-8)
  (= (road-length city-2-loc-7 city-2-loc-8) 13)
  ; 363,0 <-> 906,1
  (road city-1-loc-8 city-2-loc-3)
  (= (road-length city-1-loc-8 city-2-loc-3) 55)
  (road city-2-loc-3 city-1-loc-8)
  (= (road-length city-2-loc-3 city-1-loc-8) 55)
  (at package-1 city-1-loc-2)
  (at package-2 city-1-loc-8)
  (at package-3 city-1-loc-2)
  (at package-4 city-1-loc-6)
  (at package-5 city-1-loc-5)
  (at package-6 city-1-loc-3)
  (at truck-1 city-2-loc-4)
  (capacity truck-1 capacity-4)
  (at truck-2 city-2-loc-8)
  (capacity truck-2 capacity-4)
  (at truck-3 city-2-loc-3)
  (capacity truck-3 capacity-3)
 )
 (:goal (and
  (at package-1 city-2-loc-4)
  (at package-2 city-2-loc-5)
  (at package-3 city-2-loc-3)
  (at package-4 city-2-loc-8)
  (at package-5 city-2-loc-6)
  (at package-6 city-2-loc-6)
 ))
 (:metric minimize (total-cost))
)
