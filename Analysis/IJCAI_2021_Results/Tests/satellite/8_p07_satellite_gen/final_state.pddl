(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
satellite1 - satellite
instrument1 - instrument
satellite2 - satellite
instrument2 - instrument
instrument3 - instrument
instrument4 - instrument
instrument5 - instrument
instrument6 - instrument
satellite3 - satellite
instrument7 - instrument
instrument8 - instrument
instrument9 - instrument
satellite4 - satellite
instrument10 - instrument
instrument11 - instrument
instrument12 - instrument
instrument13 - instrument
thermograph0 - mode
thermograph3 - mode
image1 - mode
thermograph2 - mode
groundstation1 - direction
groundstation0 - direction
groundstation4 - direction
groundstation3 - direction
star2 - direction
planet5 - direction
star6 - direction
star7 - direction
planet8 - direction
star9 - direction
planet10 - direction
planet11 - direction
phenomenon12 - direction
phenomenon13 - direction
)
	(:init
			(calibration_target instrument0 star2)
			(calibration_target instrument1 star2)
			(calibration_target instrument10 groundstation0)
			(calibration_target instrument11 groundstation4)
			(calibration_target instrument12 groundstation3)
			(calibration_target instrument13 star2)
			(calibration_target instrument2 star2)
			(calibration_target instrument3 star2)
			(calibration_target instrument4 groundstation0)
			(calibration_target instrument5 star2)
			(calibration_target instrument6 groundstation1)
			(calibration_target instrument7 groundstation4)
			(calibration_target instrument8 groundstation0)
			(calibration_target instrument9 groundstation0)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite1)
			(on_board instrument10 satellite4)
			(on_board instrument11 satellite4)
			(on_board instrument12 satellite4)
			(on_board instrument13 satellite4)
			(on_board instrument2 satellite2)
			(on_board instrument3 satellite2)
			(on_board instrument4 satellite2)
			(on_board instrument5 satellite2)
			(on_board instrument6 satellite2)
			(on_board instrument7 satellite3)
			(on_board instrument8 satellite3)
			(on_board instrument9 satellite3)
			(pointing satellite0 star2)
			(pointing satellite1 groundstation4)
			(pointing satellite2 star6)
			(pointing satellite3 groundstation1)
			(pointing satellite4 groundstation4)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_avail satellite2)
			(power_avail satellite3)
			(power_avail satellite4)
			(supports instrument0 thermograph2)
			(supports instrument1 image1)
			(supports instrument1 thermograph0)
			(supports instrument1 thermograph3)
			(supports instrument10 image1)
			(supports instrument10 thermograph2)
			(supports instrument11 thermograph3)
			(supports instrument12 thermograph0)
			(supports instrument12 thermograph3)
			(supports instrument13 image1)
			(supports instrument13 thermograph2)
			(supports instrument13 thermograph3)
			(supports instrument2 image1)
			(supports instrument2 thermograph2)
			(supports instrument2 thermograph3)
			(supports instrument3 image1)
			(supports instrument3 thermograph0)
			(supports instrument3 thermograph3)
			(supports instrument4 image1)
			(supports instrument4 thermograph0)
			(supports instrument5 thermograph0)
			(supports instrument5 thermograph2)
			(supports instrument5 thermograph3)
			(supports instrument6 thermograph3)
			(supports instrument7 thermograph0)
			(supports instrument7 thermograph2)
			(supports instrument8 image1)
			(supports instrument9 thermograph3)
	)
(:goal (and
(pointing satellite1 groundstation1)
(pointing satellite4 star9)
(have_image planet5 image1)
(have_image star6 image1)
(have_image star7 image1)
(have_image planet8 thermograph3)
(have_image star9 thermograph0)
(have_image planet10 thermograph2)
(have_image planet11 thermograph0)
(have_image phenomenon12 thermograph3)
(have_image phenomenon13 thermograph0)
))
)

