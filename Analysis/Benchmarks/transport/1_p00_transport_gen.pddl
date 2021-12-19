; Transport city-sequential-8nodes-400size-4degree-10mindistance-3trucks-6packages-3529741seed

(define (problem transport-city-sequential-8nodes-400size-4degree-10mindistance-3trucks-6packages-3529741seed)
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
  ; 152,35 -> 109,127
  (road city-loc-3 city-loc-2)
  (= (road-length city-loc-3 city-loc-2) 11)
  ; 109,127 -> 152,35
  (road city-loc-2 city-loc-3)
  (= (road-length city-loc-2 city-loc-3) 11)
  ; 137,237 -> 34,397
  (road city-loc-4 city-loc-1)
  (= (road-length city-loc-4 city-loc-1) 19)
  ; 34,397 -> 137,237
  (road city-loc-1 city-loc-4)
  (= (road-length city-loc-1 city-loc-4) 19)
  ; 137,237 -> 109,127
  (road city-loc-4 city-loc-2)
  (= (road-length city-loc-4 city-loc-2) 12)
  ; 109,127 -> 137,237
  (road city-loc-2 city-loc-4)
  (= (road-length city-loc-2 city-loc-4) 12)
  ; 27,172 -> 109,127
  (road city-loc-5 city-loc-2)
  (= (road-length city-loc-5 city-loc-2) 10)
  ; 109,127 -> 27,172
  (road city-loc-2 city-loc-5)
  (= (road-length city-loc-2 city-loc-5) 10)
  ; 27,172 -> 152,35
  (road city-loc-5 city-loc-3)
  (= (road-length city-loc-5 city-loc-3) 19)
  ; 152,35 -> 27,172
  (road city-loc-3 city-loc-5)
  (= (road-length city-loc-3 city-loc-5) 19)
  ; 27,172 -> 137,237
  (road city-loc-5 city-loc-4)
  (= (road-length city-loc-5 city-loc-4) 13)
  ; 137,237 -> 27,172
  (road city-loc-4 city-loc-5)
  (= (road-length city-loc-4 city-loc-5) 13)
  ; 172,213 -> 109,127
  (road city-loc-6 city-loc-2)
  (= (road-length city-loc-6 city-loc-2) 11)
  ; 109,127 -> 172,213
  (road city-loc-2 city-loc-6)
  (= (road-length city-loc-2 city-loc-6) 11)
  ; 172,213 -> 152,35
  (road city-loc-6 city-loc-3)
  (= (road-length city-loc-6 city-loc-3) 18)
  ; 152,35 -> 172,213
  (road city-loc-3 city-loc-6)
  (= (road-length city-loc-3 city-loc-6) 18)
  ; 172,213 -> 137,237
  (road city-loc-6 city-loc-4)
  (= (road-length city-loc-6 city-loc-4) 5)
  ; 137,237 -> 172,213
  (road city-loc-4 city-loc-6)
  (= (road-length city-loc-4 city-loc-6) 5)
  ; 172,213 -> 27,172
  (road city-loc-6 city-loc-5)
  (= (road-length city-loc-6 city-loc-5) 16)
  ; 27,172 -> 172,213
  (road city-loc-5 city-loc-6)
  (= (road-length city-loc-5 city-loc-6) 16)
  ; 349,231 -> 172,213
  (road city-loc-7 city-loc-6)
  (= (road-length city-loc-7 city-loc-6) 18)
  ; 172,213 -> 349,231
  (road city-loc-6 city-loc-7)
  (= (road-length city-loc-6 city-loc-7) 18)
  ; 104,244 -> 34,397
  (road city-loc-8 city-loc-1)
  (= (road-length city-loc-8 city-loc-1) 17)
  ; 34,397 -> 104,244
  (road city-loc-1 city-loc-8)
  (= (road-length city-loc-1 city-loc-8) 17)
  ; 104,244 -> 109,127
  (road city-loc-8 city-loc-2)
  (= (road-length city-loc-8 city-loc-2) 12)
  ; 109,127 -> 104,244
  (road city-loc-2 city-loc-8)
  (= (road-length city-loc-2 city-loc-8) 12)
  ; 104,244 -> 137,237
  (road city-loc-8 city-loc-4)
  (= (road-length city-loc-8 city-loc-4) 4)
  ; 137,237 -> 104,244
  (road city-loc-4 city-loc-8)
  (= (road-length city-loc-4 city-loc-8) 4)
  ; 104,244 -> 27,172
  (road city-loc-8 city-loc-5)
  (= (road-length city-loc-8 city-loc-5) 11)
  ; 27,172 -> 104,244
  (road city-loc-5 city-loc-8)
  (= (road-length city-loc-5 city-loc-8) 11)
  ; 104,244 -> 172,213
  (road city-loc-8 city-loc-6)
  (= (road-length city-loc-8 city-loc-6) 8)
  ; 172,213 -> 104,244
  (road city-loc-6 city-loc-8)
  (= (road-length city-loc-6 city-loc-8) 8)
  (at package-1 city-loc-3)
  (at package-2 city-loc-4)
  (at package-3 city-loc-6)
  (at package-4 city-loc-5)
  (at package-5 city-loc-2)
  (at package-6 city-loc-8)
  (at truck-1 city-loc-7)
  (capacity truck-1 capacity-4)
  (at truck-2 city-loc-4)
  (capacity truck-2 capacity-4)
  (at truck-3 city-loc-8)
  (capacity truck-3 capacity-4)
 )
 (:goal (and
  (at package-1 city-loc-1)
  (at package-2 city-loc-7)
  (at package-3 city-loc-1)
  (at package-4 city-loc-7)
  (at package-5 city-loc-7)
  (at package-6 city-loc-5)
 ))
 (:metric minimize (total-cost))
)
