(define (problem dlog-2-1-3)
(:domain driverlog)
(:objects
driver1 - driver
driver2 - driver
truck1 - truck
package1 - obj
package2 - obj
package3 - obj
s0 - location
s1 - location
p0-1 - location
p1-0 - location
)
	(:init
			(at driver1 p0-1)
			(at package1 s1)
			(at package2 s0)
			(at truck1 s0)
			(driving driver2 truck1)
			(in package3 truck1)
			(link s0 s1)
			(link s1 s0)
			(path p0-1 s0)
			(path p0-1 s1)
			(path s0 p0-1)
			(path s1 p0-1)
	)
(:goal (and
(at driver1 s0)
(at driver2 s1)
(at truck1 s0)
(at package1 s0)
(at package2 s1)
(at package3 s0)
))
(:metric minimize (total-time))
)









