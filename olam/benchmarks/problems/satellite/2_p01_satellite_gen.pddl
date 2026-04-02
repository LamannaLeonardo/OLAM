(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph0 - mode
	Star0 - direction
	GroundStation1 - direction
	Phenomenon2 - direction
	Star3 - direction
	Star4 - direction
)
(:init
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 GroundStation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Phenomenon2)
)
(:goal (and
	(have_image Phenomenon2 thermograph0)
	(have_image Star3 thermograph0)
	(have_image Star4 thermograph0)
))

)
