(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
instrument1 - instrument
satellite1 - satellite
instrument2 - instrument
thermograph0 - mode
groundstation1 - direction
star0 - direction
phenomenon2 - direction
star3 - direction
)
	(:init
			(calibrated instrument0)
			(calibration_target instrument0 groundstation1)
			(calibration_target instrument1 groundstation1)
			(calibration_target instrument2 star0)
			(have_image groundstation1 thermograph0)
			(have_image star3 thermograph0)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite0)
			(on_board instrument2 satellite1)
			(pointing satellite0 star3)
			(pointing satellite1 groundstation1)
			(power_on instrument0)
			(power_on instrument2)
			(supports instrument0 thermograph0)
			(supports instrument1 thermograph0)
			(supports instrument2 thermograph0)
	)
(:goal (and
(pointing satellite0 groundstation1)
(have_image phenomenon2 thermograph0)
(have_image star3 thermograph0)
))
)















