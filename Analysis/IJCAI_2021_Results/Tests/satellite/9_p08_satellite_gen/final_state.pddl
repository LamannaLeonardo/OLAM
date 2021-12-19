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
instrument6 - instrument
instrument7 - instrument
instrument8 - instrument
instrument9 - instrument
satellite3 - satellite
instrument10 - instrument
instrument11 - instrument
instrument12 - instrument
satellite4 - satellite
instrument13 - instrument
satellite5 - satellite
instrument14 - instrument
instrument15 - instrument
thermograph3 - mode
image1 - mode
thermograph0 - mode
thermograph4 - mode
thermograph2 - mode
star4 - direction
star1 - direction
groundstation2 - direction
groundstation5 - direction
star3 - direction
star0 - direction
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
)
	(:init
			(calibration_target instrument0 star3)
			(calibration_target instrument1 star1)
			(calibration_target instrument10 groundstation2)
			(calibration_target instrument11 star0)
			(calibration_target instrument12 star3)
			(calibration_target instrument13 groundstation2)
			(calibration_target instrument13 star1)
			(calibration_target instrument14 groundstation5)
			(calibration_target instrument15 star0)
			(calibration_target instrument15 star3)
			(calibration_target instrument2 groundstation5)
			(calibration_target instrument2 star0)
			(calibration_target instrument3 groundstation2)
			(calibration_target instrument3 star1)
			(calibration_target instrument4 star3)
			(calibration_target instrument4 star4)
			(calibration_target instrument5 groundstation2)
			(calibration_target instrument5 star1)
			(calibration_target instrument6 star3)
			(calibration_target instrument6 star4)
			(calibration_target instrument7 groundstation2)
			(calibration_target instrument7 star0)
			(calibration_target instrument8 star0)
			(calibration_target instrument8 star3)
			(calibration_target instrument9 star1)
			(calibration_target instrument9 star3)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite1)
			(on_board instrument10 satellite3)
			(on_board instrument11 satellite3)
			(on_board instrument12 satellite3)
			(on_board instrument13 satellite4)
			(on_board instrument14 satellite5)
			(on_board instrument15 satellite5)
			(on_board instrument2 satellite1)
			(on_board instrument3 satellite1)
			(on_board instrument4 satellite1)
			(on_board instrument5 satellite2)
			(on_board instrument6 satellite2)
			(on_board instrument7 satellite2)
			(on_board instrument8 satellite2)
			(on_board instrument9 satellite2)
			(pointing satellite0 star0)
			(pointing satellite1 star7)
			(pointing satellite2 planet14)
			(pointing satellite3 star7)
			(pointing satellite4 phenomenon12)
			(pointing satellite5 phenomenon12)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_avail satellite2)
			(power_avail satellite3)
			(power_avail satellite4)
			(power_avail satellite5)
			(supports instrument0 thermograph0)
			(supports instrument0 thermograph4)
			(supports instrument1 thermograph2)
			(supports instrument1 thermograph4)
			(supports instrument10 thermograph0)
			(supports instrument11 thermograph2)
			(supports instrument12 thermograph0)
			(supports instrument13 thermograph0)
			(supports instrument13 thermograph4)
			(supports instrument14 thermograph2)
			(supports instrument14 thermograph4)
			(supports instrument15 thermograph2)
			(supports instrument2 image1)
			(supports instrument2 thermograph4)
			(supports instrument3 thermograph0)
			(supports instrument3 thermograph3)
			(supports instrument3 thermograph4)
			(supports instrument4 thermograph4)
			(supports instrument5 thermograph0)
			(supports instrument5 thermograph2)
			(supports instrument5 thermograph3)
			(supports instrument6 image1)
			(supports instrument6 thermograph0)
			(supports instrument6 thermograph2)
			(supports instrument7 thermograph4)
			(supports instrument8 thermograph2)
			(supports instrument9 thermograph4)
	)
(:goal (and
(pointing satellite0 planet9)
(pointing satellite2 groundstation5)
(pointing satellite4 phenomenon12)
(pointing satellite5 groundstation5)
(have_image phenomenon6 thermograph0)
(have_image star7 thermograph2)
(have_image planet9 thermograph4)
(have_image star10 thermograph2)
(have_image phenomenon12 thermograph0)
(have_image planet13 image1)
(have_image planet15 thermograph4)
))
)

