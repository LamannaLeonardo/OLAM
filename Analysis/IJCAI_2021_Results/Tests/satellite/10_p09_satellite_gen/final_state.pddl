(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
satellite1 - satellite
instrument1 - instrument
instrument2 - instrument
instrument3 - instrument
instrument4 - instrument
satellite2 - satellite
instrument5 - instrument
satellite3 - satellite
instrument6 - instrument
instrument7 - instrument
instrument8 - instrument
satellite4 - satellite
instrument9 - instrument
instrument10 - instrument
satellite5 - satellite
instrument11 - instrument
instrument12 - instrument
instrument13 - instrument
instrument14 - instrument
instrument15 - instrument
thermograph4 - mode
thermograph3 - mode
image1 - mode
thermograph2 - mode
thermograph0 - mode
groundstation5 - direction
star4 - direction
star1 - direction
star3 - direction
star0 - direction
groundstation2 - direction
phenomenon6 - direction
star7 - direction
star8 - direction
planet9 - direction
star10 - direction
star11 - direction
phenomenon12 - direction
planet13 - direction
planet14 - direction
planet15 - direction
star16 - direction
)
	(:init
			(calibration_target instrument0 star0)
			(calibration_target instrument0 star1)
			(calibration_target instrument1 groundstation2)
			(calibration_target instrument1 star4)
			(calibration_target instrument10 groundstation2)
			(calibration_target instrument10 star0)
			(calibration_target instrument11 groundstation2)
			(calibration_target instrument11 star4)
			(calibration_target instrument12 groundstation2)
			(calibration_target instrument12 star1)
			(calibration_target instrument13 groundstation2)
			(calibration_target instrument13 star3)
			(calibration_target instrument14 star0)
			(calibration_target instrument14 star3)
			(calibration_target instrument15 groundstation2)
			(calibration_target instrument2 groundstation5)
			(calibration_target instrument2 star3)
			(calibration_target instrument3 groundstation2)
			(calibration_target instrument4 star0)
			(calibration_target instrument5 star4)
			(calibration_target instrument6 groundstation2)
			(calibration_target instrument7 star0)
			(calibration_target instrument8 star3)
			(calibration_target instrument8 star4)
			(calibration_target instrument9 star1)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite1)
			(on_board instrument10 satellite4)
			(on_board instrument11 satellite5)
			(on_board instrument12 satellite5)
			(on_board instrument13 satellite5)
			(on_board instrument14 satellite5)
			(on_board instrument15 satellite5)
			(on_board instrument2 satellite1)
			(on_board instrument3 satellite1)
			(on_board instrument4 satellite1)
			(on_board instrument5 satellite2)
			(on_board instrument6 satellite3)
			(on_board instrument7 satellite3)
			(on_board instrument8 satellite3)
			(on_board instrument9 satellite4)
			(pointing satellite0 planet13)
			(pointing satellite1 planet13)
			(pointing satellite2 planet15)
			(pointing satellite3 star3)
			(pointing satellite4 planet13)
			(pointing satellite5 planet9)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_avail satellite2)
			(power_avail satellite3)
			(power_avail satellite4)
			(power_avail satellite5)
			(supports instrument0 thermograph0)
			(supports instrument0 thermograph4)
			(supports instrument1 image1)
			(supports instrument1 thermograph2)
			(supports instrument10 thermograph2)
			(supports instrument11 image1)
			(supports instrument11 thermograph0)
			(supports instrument11 thermograph2)
			(supports instrument12 thermograph2)
			(supports instrument13 thermograph0)
			(supports instrument13 thermograph2)
			(supports instrument13 thermograph3)
			(supports instrument14 thermograph0)
			(supports instrument14 thermograph2)
			(supports instrument15 image1)
			(supports instrument15 thermograph0)
			(supports instrument15 thermograph2)
			(supports instrument2 thermograph0)
			(supports instrument2 thermograph3)
			(supports instrument3 thermograph0)
			(supports instrument3 thermograph2)
			(supports instrument4 thermograph2)
			(supports instrument4 thermograph3)
			(supports instrument5 thermograph0)
			(supports instrument5 thermograph3)
			(supports instrument5 thermograph4)
			(supports instrument6 thermograph0)
			(supports instrument6 thermograph2)
			(supports instrument7 thermograph0)
			(supports instrument7 thermograph2)
			(supports instrument7 thermograph3)
			(supports instrument8 thermograph0)
			(supports instrument8 thermograph3)
			(supports instrument9 thermograph2)
			(supports instrument9 thermograph3)
	)
(:goal (and
(pointing satellite2 star4)
(pointing satellite4 star1)
(pointing satellite5 star8)
(have_image phenomenon6 thermograph0)
(have_image star7 thermograph2)
(have_image planet9 thermograph4)
(have_image star10 thermograph2)
(have_image phenomenon12 thermograph0)
(have_image planet13 image1)
(have_image planet15 thermograph4)
(have_image star16 thermograph2)
))
)

