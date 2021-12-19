(define (problem dlog-2-2-3)
(:domain driverlog)
(:objects
driver1 - driver
driver2 - driver
truck1 - truck
truck2 - truck
package1 - obj
package2 - obj
package3 - obj
s0 - location
)
	(:init
			(at driver2 s0)
			(at package3 s0)
			(at truck1 s0)
			(at truck2 s0)
			(driving driver1 truck1)
			(empty truck2)
			(in package1 truck1)
			(in package2 truck1)
	)
(:goal (and
(at driver1 s0)
(at driver2 s0)
(at truck2 s0)
(at package1 s0)
(at package2 s0)
(at package3 s0)
))
(:metric minimize (total-time))
)










