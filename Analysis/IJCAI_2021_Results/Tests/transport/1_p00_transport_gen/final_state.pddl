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
			(at package-1 city-loc-3)
			(at package-2 city-loc-4)
			(at package-3 city-loc-6)
			(at package-5 city-loc-2)
			(at package-6 city-loc-8)
			(at truck-1 city-loc-7)
			(at truck-2 city-loc-4)
			(at truck-3 city-loc-5)
			(capacity truck-1 capacity-4)
			(capacity truck-2 capacity-4)
			(capacity truck-3 capacity-3)
			(capacity-predecessor capacity-0 capacity-1)
			(capacity-predecessor capacity-1 capacity-2)
			(capacity-predecessor capacity-2 capacity-3)
			(capacity-predecessor capacity-3 capacity-4)
			(in package-4 truck-3)
			(road city-loc-1 city-loc-4)
			(road city-loc-1 city-loc-8)
			(road city-loc-2 city-loc-3)
			(road city-loc-2 city-loc-4)
			(road city-loc-2 city-loc-5)
			(road city-loc-2 city-loc-6)
			(road city-loc-2 city-loc-8)
			(road city-loc-3 city-loc-2)
			(road city-loc-3 city-loc-5)
			(road city-loc-3 city-loc-6)
			(road city-loc-4 city-loc-1)
			(road city-loc-4 city-loc-2)
			(road city-loc-4 city-loc-5)
			(road city-loc-4 city-loc-6)
			(road city-loc-4 city-loc-8)
			(road city-loc-5 city-loc-2)
			(road city-loc-5 city-loc-3)
			(road city-loc-5 city-loc-4)
			(road city-loc-5 city-loc-6)
			(road city-loc-5 city-loc-8)
			(road city-loc-6 city-loc-2)
			(road city-loc-6 city-loc-3)
			(road city-loc-6 city-loc-4)
			(road city-loc-6 city-loc-5)
			(road city-loc-6 city-loc-7)
			(road city-loc-6 city-loc-8)
			(road city-loc-7 city-loc-6)
			(road city-loc-8 city-loc-1)
			(road city-loc-8 city-loc-2)
			(road city-loc-8 city-loc-4)
			(road city-loc-8 city-loc-5)
			(road city-loc-8 city-loc-6)
	)
(:goal (and
(at package-1 city-loc-1)
(at package-2 city-loc-7)
(at package-3 city-loc-1)
(at package-4 city-loc-7)
(at package-5 city-loc-7)
(at package-6 city-loc-5)
))
)





