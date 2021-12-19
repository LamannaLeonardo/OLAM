(define (problem roverprob0) (:domain rover)
(:objects
general - lander
colour high_res low_res - mode
rover0 - rover
rover0store - store
waypoint0 waypoint1 waypoint2 - waypoint
camera0 - camera
objective0 - objective
)
	(:init
			(at rover0 waypoint1)
			(at_lander general waypoint2)
			(available rover0)
			(calibration_target camera0 objective0)
			(can_traverse rover0 waypoint0 waypoint1)
			(can_traverse rover0 waypoint1 waypoint0)
			(can_traverse rover0 waypoint1 waypoint2)
			(can_traverse rover0 waypoint2 waypoint1)
			(channel_free general)
			(communicated_image_data objective0 low_res)
			(communicated_rock_data waypoint0)
			(communicated_rock_data waypoint1)
			(communicated_rock_data waypoint2)
			(communicated_soil_data waypoint1)
			(communicated_soil_data waypoint2)
			(empty rover0store)
			(equipped_for_imaging rover0)
			(equipped_for_rock_analysis rover0)
			(equipped_for_soil_analysis rover0)
			(have_image rover0 objective0 low_res)
			(have_rock_analysis rover0 waypoint0)
			(have_rock_analysis rover0 waypoint1)
			(have_rock_analysis rover0 waypoint2)
			(have_soil_analysis rover0 waypoint1)
			(have_soil_analysis rover0 waypoint2)
			(on_board camera0 rover0)
			(store_of rover0store rover0)
			(supports camera0 low_res)
			(visible waypoint0 waypoint1)
			(visible waypoint0 waypoint2)
			(visible waypoint1 waypoint0)
			(visible waypoint1 waypoint2)
			(visible waypoint2 waypoint0)
			(visible waypoint2 waypoint1)
			(visible_from objective0 waypoint0)
			(visible_from objective0 waypoint1)
			(visible_from objective0 waypoint2)
	)
(:goal (and
(communicated_soil_data waypoint1)
(communicated_soil_data waypoint2)
(communicated_rock_data waypoint2)
(communicated_rock_data waypoint0)
(communicated_image_data objective0 low_res)
)
)
)























































