; Transport city-sequential-12nodes-400size-4degree-10mindistance-3trucks-12packages-3405526seed

(define (problem transport-city-sequential-12nodes-400size-4degree-10mindistance-3trucks-12packages-3405526seed)
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
  city-loc-11 - location
  city-loc-12 - location
  truck-1 - vehicle
  truck-2 - vehicle
  truck-3 - vehicle
  package-1 - package
  package-2 - package
  package-3 - package
  package-4 - package
  package-5 - package
  package-6 - package
  package-7 - package
  package-8 - package
  package-9 - package
  package-10 - package
  package-11 - package
  package-12 - package
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
  ; 369,249 -> 328,350
  (road city-loc-4 city-loc-3)
  (= (road-length city-loc-4 city-loc-3) 11)
  ; 328,350 -> 369,249
  (road city-loc-3 city-loc-4)
  (= (road-length city-loc-3 city-loc-4) 11)
  ; 267,3 -> 169,85
  (road city-loc-5 city-loc-1)
  (= (road-length city-loc-5 city-loc-1) 13)
  ; 169,85 -> 267,3
  (road city-loc-1 city-loc-5)
  (= (road-length city-loc-1 city-loc-5) 13)
  ; 229,332 -> 166,384
  (road city-loc-6 city-loc-2)
  (= (road-length city-loc-6 city-loc-2) 9)
  ; 166,384 -> 229,332
  (road city-loc-2 city-loc-6)
  (= (road-length city-loc-2 city-loc-6) 9)
  ; 229,332 -> 328,350
  (road city-loc-6 city-loc-3)
  (= (road-length city-loc-6 city-loc-3) 11)
  ; 328,350 -> 229,332
  (road city-loc-3 city-loc-6)
  (= (road-length city-loc-3 city-loc-6) 11)
  ; 79,363 -> 166,384
  (road city-loc-7 city-loc-2)
  (= (road-length city-loc-7 city-loc-2) 9)
  ; 166,384 -> 79,363
  (road city-loc-2 city-loc-7)
  (= (road-length city-loc-2 city-loc-7) 9)
  ; 79,363 -> 229,332
  (road city-loc-7 city-loc-6)
  (= (road-length city-loc-7 city-loc-6) 16)
  ; 229,332 -> 79,363
  (road city-loc-6 city-loc-7)
  (= (road-length city-loc-6 city-loc-7) 16)
  ; 276,343 -> 166,384
  (road city-loc-8 city-loc-2)
  (= (road-length city-loc-8 city-loc-2) 12)
  ; 166,384 -> 276,343
  (road city-loc-2 city-loc-8)
  (= (road-length city-loc-2 city-loc-8) 12)
  ; 276,343 -> 328,350
  (road city-loc-8 city-loc-3)
  (= (road-length city-loc-8 city-loc-3) 6)
  ; 328,350 -> 276,343
  (road city-loc-3 city-loc-8)
  (= (road-length city-loc-3 city-loc-8) 6)
  ; 276,343 -> 369,249
  (road city-loc-8 city-loc-4)
  (= (road-length city-loc-8 city-loc-4) 14)
  ; 369,249 -> 276,343
  (road city-loc-4 city-loc-8)
  (= (road-length city-loc-4 city-loc-8) 14)
  ; 276,343 -> 229,332
  (road city-loc-8 city-loc-6)
  (= (road-length city-loc-8 city-loc-6) 5)
  ; 229,332 -> 276,343
  (road city-loc-6 city-loc-8)
  (= (road-length city-loc-6 city-loc-8) 5)
  ; 179,252 -> 166,384
  (road city-loc-9 city-loc-2)
  (= (road-length city-loc-9 city-loc-2) 14)
  ; 166,384 -> 179,252
  (road city-loc-2 city-loc-9)
  (= (road-length city-loc-2 city-loc-9) 14)
  ; 179,252 -> 229,332
  (road city-loc-9 city-loc-6)
  (= (road-length city-loc-9 city-loc-6) 10)
  ; 229,332 -> 179,252
  (road city-loc-6 city-loc-9)
  (= (road-length city-loc-6 city-loc-9) 10)
  ; 179,252 -> 79,363
  (road city-loc-9 city-loc-7)
  (= (road-length city-loc-9 city-loc-7) 15)
  ; 79,363 -> 179,252
  (road city-loc-7 city-loc-9)
  (= (road-length city-loc-7 city-loc-9) 15)
  ; 179,252 -> 276,343
  (road city-loc-9 city-loc-8)
  (= (road-length city-loc-9 city-loc-8) 14)
  ; 276,343 -> 179,252
  (road city-loc-8 city-loc-9)
  (= (road-length city-loc-8 city-loc-9) 14)
  ; 216,186 -> 169,85
  (road city-loc-10 city-loc-1)
  (= (road-length city-loc-10 city-loc-1) 12)
  ; 169,85 -> 216,186
  (road city-loc-1 city-loc-10)
  (= (road-length city-loc-1 city-loc-10) 12)
  ; 216,186 -> 229,332
  (road city-loc-10 city-loc-6)
  (= (road-length city-loc-10 city-loc-6) 15)
  ; 229,332 -> 216,186
  (road city-loc-6 city-loc-10)
  (= (road-length city-loc-6 city-loc-10) 15)
  ; 216,186 -> 179,252
  (road city-loc-10 city-loc-9)
  (= (road-length city-loc-10 city-loc-9) 8)
  ; 179,252 -> 216,186
  (road city-loc-9 city-loc-10)
  (= (road-length city-loc-9 city-loc-10) 8)
  ; 302,225 -> 328,350
  (road city-loc-11 city-loc-3)
  (= (road-length city-loc-11 city-loc-3) 13)
  ; 328,350 -> 302,225
  (road city-loc-3 city-loc-11)
  (= (road-length city-loc-3 city-loc-11) 13)
  ; 302,225 -> 369,249
  (road city-loc-11 city-loc-4)
  (= (road-length city-loc-11 city-loc-4) 8)
  ; 369,249 -> 302,225
  (road city-loc-4 city-loc-11)
  (= (road-length city-loc-4 city-loc-11) 8)
  ; 302,225 -> 229,332
  (road city-loc-11 city-loc-6)
  (= (road-length city-loc-11 city-loc-6) 13)
  ; 229,332 -> 302,225
  (road city-loc-6 city-loc-11)
  (= (road-length city-loc-6 city-loc-11) 13)
  ; 302,225 -> 276,343
  (road city-loc-11 city-loc-8)
  (= (road-length city-loc-11 city-loc-8) 13)
  ; 276,343 -> 302,225
  (road city-loc-8 city-loc-11)
  (= (road-length city-loc-8 city-loc-11) 13)
  ; 302,225 -> 179,252
  (road city-loc-11 city-loc-9)
  (= (road-length city-loc-11 city-loc-9) 13)
  ; 179,252 -> 302,225
  (road city-loc-9 city-loc-11)
  (= (road-length city-loc-9 city-loc-11) 13)
  ; 302,225 -> 216,186
  (road city-loc-11 city-loc-10)
  (= (road-length city-loc-11 city-loc-10) 10)
  ; 216,186 -> 302,225
  (road city-loc-10 city-loc-11)
  (= (road-length city-loc-10 city-loc-11) 10)
  ; 257,368 -> 166,384
  (road city-loc-12 city-loc-2)
  (= (road-length city-loc-12 city-loc-2) 10)
  ; 166,384 -> 257,368
  (road city-loc-2 city-loc-12)
  (= (road-length city-loc-2 city-loc-12) 10)
  ; 257,368 -> 328,350
  (road city-loc-12 city-loc-3)
  (= (road-length city-loc-12 city-loc-3) 8)
  ; 328,350 -> 257,368
  (road city-loc-3 city-loc-12)
  (= (road-length city-loc-3 city-loc-12) 8)
  ; 257,368 -> 229,332
  (road city-loc-12 city-loc-6)
  (= (road-length city-loc-12 city-loc-6) 5)
  ; 229,332 -> 257,368
  (road city-loc-6 city-loc-12)
  (= (road-length city-loc-6 city-loc-12) 5)
  ; 257,368 -> 276,343
  (road city-loc-12 city-loc-8)
  (= (road-length city-loc-12 city-loc-8) 4)
  ; 276,343 -> 257,368
  (road city-loc-8 city-loc-12)
  (= (road-length city-loc-8 city-loc-12) 4)
  ; 257,368 -> 179,252
  (road city-loc-12 city-loc-9)
  (= (road-length city-loc-12 city-loc-9) 14)
  ; 179,252 -> 257,368
  (road city-loc-9 city-loc-12)
  (= (road-length city-loc-9 city-loc-12) 14)
  ; 257,368 -> 302,225
  (road city-loc-12 city-loc-11)
  (= (road-length city-loc-12 city-loc-11) 15)
  ; 302,225 -> 257,368
  (road city-loc-11 city-loc-12)
  (= (road-length city-loc-11 city-loc-12) 15)
  (at package-1 city-loc-5)
  (at package-2 city-loc-5)
  (at package-3 city-loc-9)
  (at package-4 city-loc-9)
  (at package-5 city-loc-4)
  (at package-6 city-loc-12)
  (at package-7 city-loc-4)
  (at package-8 city-loc-2)
  (at package-9 city-loc-3)
  (at package-10 city-loc-8)
  (at package-11 city-loc-3)
  (at package-12 city-loc-2)
  (at truck-1 city-loc-1)
  (capacity truck-1 capacity-2)
  (at truck-2 city-loc-10)
  (capacity truck-2 capacity-4)
  (at truck-3 city-loc-11)
  (capacity truck-3 capacity-4)
 )
 (:goal (and
  (at package-1 city-loc-9)
  (at package-2 city-loc-10)
  (at package-3 city-loc-11)
  (at package-4 city-loc-11)
  (at package-5 city-loc-3)
  (at package-6 city-loc-4)
  (at package-7 city-loc-8)
  (at package-8 city-loc-9)
  (at package-9 city-loc-8)
  (at package-10 city-loc-5)
  (at package-11 city-loc-6)
  (at package-12 city-loc-7)
 ))
 (:metric minimize (total-cost))
)
