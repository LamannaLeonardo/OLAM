(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
instrument1 - instrument
instrument2 - instrument
satellite1 - satellite
instrument3 - instrument
satellite2 - satellite
instrument4 - instrument
instrument5 - instrument
thermograph0 - mode
image1 - mode
star1 - direction
star2 - direction
groundstation0 - direction
planet3 - direction
planet4 - direction
planet5 - direction
phenomenon6 - direction
star7 - direction
)
	(:init
			(calibrated instrument4)
			(calibration_target instrument0 star2)
			(calibration_target instrument1 star1)
			(calibration_target instrument2 star2)
			(calibration_target instrument3 star2)
			(calibration_target instrument4 groundstation0)
			(calibration_target instrument5 groundstation0)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite0)
			(on_board instrument2 satellite0)
			(on_board instrument3 satellite1)
			(on_board instrument4 satellite2)
			(on_board instrument5 satellite2)
			(pointing satellite0 phenomenon6)
			(pointing satellite1 groundstation0)
			(pointing satellite2 groundstation0)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_on instrument4)
			(supports instrument0 image1)
			(supports instrument0 thermograph0)
			(supports instrument1 image1)
			(supports instrument2 image1)
			(supports instrument3 thermograph0)
			(supports instrument4 image1)
			(supports instrument5 image1)
	)
(:goal (and
(pointing satellite1 planet5)
(pointing satellite2 planet4)
(have_image planet3 thermograph0)
(have_image planet4 thermograph0)
(have_image planet5 thermograph0)
(have_image phenomenon6 thermograph0)
(have_image star7 image1)
))
)




