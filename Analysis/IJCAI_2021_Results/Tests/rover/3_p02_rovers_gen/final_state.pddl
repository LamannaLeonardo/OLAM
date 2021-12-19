(define (problem roverprob0) (:domain rover)
(:objects
general - lander
colour high_res low_res - mode
rover0 rover1 - rover
rover0store rover1store - store
waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 - waypoint
camera0 camera1 - camera
objective0 objective1 - objective
)
	(:init
			(at rover0 waypoint0)
			(at rover1 waypoint0)
			(at_lander general waypoint0)
			(at_soil_sample waypoint3)
			(available rover0)
			(available rover1)
			(calibration_target camera0 objective1)
			(calibration_target camera1 objective1)
			(can_traverse rover0 waypoint0 waypoint1)
			(can_traverse rover0 waypoint0 waypoint3)
			(can_traverse rover0 waypoint1 waypoint0)
			(can_traverse rover0 waypoint1 waypoint4)
			(can_traverse rover0 waypoint3 waypoint0)
			(can_traverse rover0 waypoint4 waypoint1)
			(can_traverse rover1 waypoint0 waypoint3)
			(can_traverse rover1 waypoint0 waypoint4)
			(can_traverse rover1 waypoint1 waypoint4)
			(can_traverse rover1 waypoint2 waypoint4)
			(can_traverse rover1 waypoint3 waypoint0)
			(can_traverse rover1 waypoint4 waypoint0)
			(can_traverse rover1 waypoint4 waypoint1)
			(can_traverse rover1 waypoint4 waypoint2)
			(channel_free general)
			(communicated_image_data objective1 low_res)
			(communicated_rock_data waypoint4)
			(communicated_soil_data waypoint4)
			(equipped_for_imaging rover0)
			(equipped_for_imaging rover1)
			(equipped_for_rock_analysis rover1)
			(equipped_for_soil_analysis rover0)
			(full rover0store)
			(full rover1store)
			(have_image rover0 objective1 low_res)
			(have_rock_analysis rover1 waypoint0)
			(have_rock_analysis rover1 waypoint4)
			(have_soil_analysis rover0 waypoint0)
			(have_soil_analysis rover0 waypoint4)
			(on_board camera0 rover0)
			(on_board camera1 rover1)
			(store_of rover0store rover0)
			(store_of rover1store rover1)
			(supports camera0 low_res)
			(supports camera1 high_res)
			(visible waypoint0 waypoint1)
			(visible waypoint0 waypoint2)
			(visible waypoint0 waypoint3)
			(visible waypoint0 waypoint4)
			(visible waypoint1 waypoint0)
			(visible waypoint1 waypoint2)
			(visible waypoint1 waypoint3)
			(visible waypoint1 waypoint4)
			(visible waypoint2 waypoint0)
			(visible waypoint2 waypoint1)
			(visible waypoint2 waypoint3)
			(visible waypoint2 waypoint4)
			(visible waypoint3 waypoint0)
			(visible waypoint3 waypoint1)
			(visible waypoint3 waypoint2)
			(visible waypoint4 waypoint0)
			(visible waypoint4 waypoint1)
			(visible waypoint4 waypoint2)
			(visible_from objective0 waypoint0)
			(visible_from objective0 waypoint1)
			(visible_from objective1 waypoint0)
			(visible_from objective1 waypoint1)
	)
(:goal (and
(communicated_soil_data waypoint0)
(communicated_soil_data waypoint4)
(communicated_soil_data waypoint3)
(communicated_rock_data waypoint0)
(communicated_rock_data waypoint4)
(communicated_image_data objective1 low_res)
)
)
)




















