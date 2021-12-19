(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
satellite0 - satellite
instrument0 - instrument
satellite1 - satellite
instrument1 - instrument
instrument2 - instrument
instrument3 - instrument
satellite2 - satellite
instrument4 - instrument
instrument5 - instrument
instrument6 - instrument
satellite3 - satellite
instrument7 - instrument
instrument8 - instrument
thermograph2 - mode
thermograph0 - mode
image1 - mode
groundstation0 - direction
star3 - direction
star1 - direction
groundstation2 - direction
star4 - direction
star5 - direction
planet6 - direction
planet7 - direction
phenomenon8 - direction
phenomenon9 - direction
planet10 - direction
)
	(:init
			(calibration_target instrument0 star3)
			(calibration_target instrument1 groundstation2)
			(calibration_target instrument2 groundstation2)
			(calibration_target instrument3 star3)
			(calibration_target instrument4 star1)
			(calibration_target instrument5 star3)
			(calibration_target instrument6 groundstation2)
			(calibration_target instrument7 star1)
			(calibration_target instrument8 groundstation2)
			(on_board instrument0 satellite0)
			(on_board instrument1 satellite1)
			(on_board instrument2 satellite1)
			(on_board instrument3 satellite1)
			(on_board instrument4 satellite2)
			(on_board instrument5 satellite2)
			(on_board instrument6 satellite2)
			(on_board instrument7 satellite3)
			(on_board instrument8 satellite3)
			(pointing satellite0 star4)
			(pointing satellite1 groundstation2)
			(pointing satellite2 star3)
			(pointing satellite3 planet7)
			(power_avail satellite0)
			(power_avail satellite1)
			(power_avail satellite2)
			(power_avail satellite3)
			(supports instrument0 image1)
			(supports instrument0 thermograph0)
			(supports instrument0 thermograph2)
			(supports instrument1 image1)
			(supports instrument1 thermograph0)
			(supports instrument1 thermograph2)
			(supports instrument2 thermograph0)
			(supports instrument3 image1)
			(supports instrument3 thermograph0)
			(supports instrument3 thermograph2)
			(supports instrument4 thermograph0)
			(supports instrument5 thermograph2)
			(supports instrument6 image1)
			(supports instrument6 thermograph0)
			(supports instrument6 thermograph2)
			(supports instrument7 image1)
			(supports instrument7 thermograph0)
			(supports instrument8 image1)
	)
(:goal (and
(have_image star5 image1)
(have_image planet6 thermograph2)
(have_image planet7 thermograph2)
(have_image phenomenon8 thermograph2)
(have_image phenomenon9 thermograph2)
(have_image planet10 thermograph0)
))
)

