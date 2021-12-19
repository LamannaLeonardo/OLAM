(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
instrument1 - instrument
instrument2 - instrument
satellite1 - satellite
instrument3 - instrument
instrument4 - instrument
instrument5 - instrument
satellite2 - satellite
instrument6 - instrument
instrument7 - instrument
instrument8 - instrument
image1 - mode
thermograph0 - mode
star2 - direction
groundstation0 - direction
star1 - direction
planet3 - direction
planet4 - direction
planet5 - direction
phenomenon6 - direction
)
	(:init
			(calibration_target instrument0 star1)
			(calibration_target instrument1 star2)
			(calibration_target instrument2 star1)
			(calibration_target instrument3 star2)
			(calibration_target instrument4 star1)
			(calibration_target instrument5 star2)
			(calibration_target instrument6 groundstation0)
			(calibration_target instrument7 star1)
			(calibration_target instrument8 star1)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite0)
			(on_board instrument2 satellite0)
			(on_board instrument3 satellite1)
			(on_board instrument4 satellite1)
			(on_board instrument5 satellite1)
			(on_board instrument6 satellite2)
			(on_board instrument7 satellite2)
			(on_board instrument8 satellite2)
			(pointing satellite0 star1)
			(pointing satellite1 star1)
			(pointing satellite2 planet3)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_avail satellite2)
			(supports instrument0 image1)
			(supports instrument0 thermograph0)
			(supports instrument1 image1)
			(supports instrument1 thermograph0)
			(supports instrument2 image1)
			(supports instrument2 thermograph0)
			(supports instrument3 image1)
			(supports instrument3 thermograph0)
			(supports instrument4 image1)
			(supports instrument5 thermograph0)
			(supports instrument6 image1)
			(supports instrument6 thermograph0)
			(supports instrument7 image1)
			(supports instrument7 thermograph0)
			(supports instrument8 thermograph0)
	)
(:goal (and
(have_image planet3 thermograph0)
(have_image planet4 thermograph0)
(have_image planet5 thermograph0)
(have_image phenomenon6 thermograph0)
))
)

