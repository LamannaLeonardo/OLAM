(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
satellite1 - satellite
instrument1 - instrument
thermograph0 - mode
star0 - direction
groundstation1 - direction
phenomenon2 - direction
star3 - direction
star4 - direction
)
	(:init
			(calibration_target instrument0 star0)
			(calibration_target instrument1 groundstation1)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite1)
			(pointing satellite0 groundstation1)
			(pointing satellite1 phenomenon2)
			(power_avail satellite0)
			(power_avail satellite1)
			(supports instrument0 thermograph0)
			(supports instrument1 thermograph0)
	)
(:goal (and
(have_image phenomenon2 thermograph0)
(have_image star3 thermograph0)
(have_image star4 thermograph0)
))
)

