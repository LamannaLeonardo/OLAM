(define (problem roverprob0) (:domain rover)
(:objects
general - lander
colour high_res low_res - mode
rover0 rover1 rover2 rover3 rover4 - rover
rover0store rover1store rover2store rover3store rover4store - store
waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 waypoint8 waypoint9 waypoint10 waypoint11 - waypoint
camera0 camera1 camera2 camera3 camera4 camera5 - camera
objective0 objective1 objective2 objective3 objective4 objective5 - objective
)
	(:init
			(at rover0 waypoint5)
			(at rover1 waypoint8)
			(at rover2 waypoint11)
			(at rover3 waypoint11)
			(at rover4 waypoint5)
			(at_lander general waypoint1)
			(at_rock_sample waypoint0)
			(at_rock_sample waypoint10)
			(at_rock_sample waypoint4)
			(at_rock_sample waypoint5)
			(at_rock_sample waypoint6)
			(at_rock_sample waypoint7)
			(at_rock_sample waypoint8)
			(at_rock_sample waypoint9)
			(at_soil_sample waypoint11)
			(at_soil_sample waypoint2)
			(at_soil_sample waypoint3)
			(at_soil_sample waypoint4)
			(at_soil_sample waypoint6)
			(available rover0)
			(available rover1)
			(available rover2)
			(available rover3)
			(available rover4)
			(calibration_target camera0 objective4)
			(calibration_target camera1 objective3)
			(calibration_target camera2 objective2)
			(calibration_target camera3 objective0)
			(calibration_target camera4 objective4)
			(calibration_target camera5 objective2)
			(can_traverse rover0 waypoint0 waypoint5)
			(can_traverse rover0 waypoint0 waypoint6)
			(can_traverse rover0 waypoint1 waypoint10)
			(can_traverse rover0 waypoint1 waypoint5)
			(can_traverse rover0 waypoint1 waypoint8)
			(can_traverse rover0 waypoint10 waypoint1)
			(can_traverse rover0 waypoint11 waypoint2)
			(can_traverse rover0 waypoint2 waypoint11)
			(can_traverse rover0 waypoint2 waypoint5)
			(can_traverse rover0 waypoint3 waypoint9)
			(can_traverse rover0 waypoint4 waypoint7)
			(can_traverse rover0 waypoint5 waypoint0)
			(can_traverse rover0 waypoint5 waypoint1)
			(can_traverse rover0 waypoint5 waypoint2)
			(can_traverse rover0 waypoint5 waypoint7)
			(can_traverse rover0 waypoint6 waypoint0)
			(can_traverse rover0 waypoint6 waypoint9)
			(can_traverse rover0 waypoint7 waypoint4)
			(can_traverse rover0 waypoint7 waypoint5)
			(can_traverse rover0 waypoint8 waypoint1)
			(can_traverse rover0 waypoint9 waypoint3)
			(can_traverse rover0 waypoint9 waypoint6)
			(can_traverse rover1 waypoint1 waypoint4)
			(can_traverse rover1 waypoint1 waypoint5)
			(can_traverse rover1 waypoint1 waypoint7)
			(can_traverse rover1 waypoint1 waypoint8)
			(can_traverse rover1 waypoint1 waypoint9)
			(can_traverse rover1 waypoint10 waypoint4)
			(can_traverse rover1 waypoint11 waypoint7)
			(can_traverse rover1 waypoint2 waypoint5)
			(can_traverse rover1 waypoint3 waypoint6)
			(can_traverse rover1 waypoint4 waypoint1)
			(can_traverse rover1 waypoint4 waypoint10)
			(can_traverse rover1 waypoint5 waypoint1)
			(can_traverse rover1 waypoint5 waypoint2)
			(can_traverse rover1 waypoint6 waypoint3)
			(can_traverse rover1 waypoint6 waypoint8)
			(can_traverse rover1 waypoint7 waypoint1)
			(can_traverse rover1 waypoint7 waypoint11)
			(can_traverse rover1 waypoint8 waypoint1)
			(can_traverse rover1 waypoint8 waypoint6)
			(can_traverse rover1 waypoint9 waypoint1)
			(can_traverse rover2 waypoint0 waypoint11)
			(can_traverse rover2 waypoint0 waypoint6)
			(can_traverse rover2 waypoint1 waypoint5)
			(can_traverse rover2 waypoint10 waypoint11)
			(can_traverse rover2 waypoint11 waypoint0)
			(can_traverse rover2 waypoint11 waypoint10)
			(can_traverse rover2 waypoint11 waypoint2)
			(can_traverse rover2 waypoint11 waypoint5)
			(can_traverse rover2 waypoint11 waypoint7)
			(can_traverse rover2 waypoint11 waypoint8)
			(can_traverse rover2 waypoint11 waypoint9)
			(can_traverse rover2 waypoint2 waypoint11)
			(can_traverse rover2 waypoint3 waypoint7)
			(can_traverse rover2 waypoint4 waypoint7)
			(can_traverse rover2 waypoint5 waypoint1)
			(can_traverse rover2 waypoint5 waypoint11)
			(can_traverse rover2 waypoint6 waypoint0)
			(can_traverse rover2 waypoint7 waypoint11)
			(can_traverse rover2 waypoint7 waypoint3)
			(can_traverse rover2 waypoint7 waypoint4)
			(can_traverse rover2 waypoint8 waypoint11)
			(can_traverse rover2 waypoint9 waypoint11)
			(can_traverse rover3 waypoint0 waypoint11)
			(can_traverse rover3 waypoint0 waypoint2)
			(can_traverse rover3 waypoint0 waypoint6)
			(can_traverse rover3 waypoint1 waypoint5)
			(can_traverse rover3 waypoint10 waypoint11)
			(can_traverse rover3 waypoint11 waypoint0)
			(can_traverse rover3 waypoint11 waypoint10)
			(can_traverse rover3 waypoint11 waypoint5)
			(can_traverse rover3 waypoint11 waypoint7)
			(can_traverse rover3 waypoint11 waypoint8)
			(can_traverse rover3 waypoint11 waypoint9)
			(can_traverse rover3 waypoint2 waypoint0)
			(can_traverse rover3 waypoint3 waypoint7)
			(can_traverse rover3 waypoint4 waypoint7)
			(can_traverse rover3 waypoint5 waypoint1)
			(can_traverse rover3 waypoint5 waypoint11)
			(can_traverse rover3 waypoint6 waypoint0)
			(can_traverse rover3 waypoint7 waypoint11)
			(can_traverse rover3 waypoint7 waypoint3)
			(can_traverse rover3 waypoint7 waypoint4)
			(can_traverse rover3 waypoint8 waypoint11)
			(can_traverse rover3 waypoint9 waypoint11)
			(can_traverse rover4 waypoint0 waypoint2)
			(can_traverse rover4 waypoint0 waypoint5)
			(can_traverse rover4 waypoint0 waypoint6)
			(can_traverse rover4 waypoint1 waypoint5)
			(can_traverse rover4 waypoint1 waypoint9)
			(can_traverse rover4 waypoint10 waypoint11)
			(can_traverse rover4 waypoint11 waypoint10)
			(can_traverse rover4 waypoint11 waypoint5)
			(can_traverse rover4 waypoint2 waypoint0)
			(can_traverse rover4 waypoint3 waypoint7)
			(can_traverse rover4 waypoint4 waypoint7)
			(can_traverse rover4 waypoint5 waypoint0)
			(can_traverse rover4 waypoint5 waypoint1)
			(can_traverse rover4 waypoint5 waypoint11)
			(can_traverse rover4 waypoint5 waypoint7)
			(can_traverse rover4 waypoint6 waypoint0)
			(can_traverse rover4 waypoint7 waypoint3)
			(can_traverse rover4 waypoint7 waypoint4)
			(can_traverse rover4 waypoint7 waypoint5)
			(can_traverse rover4 waypoint7 waypoint8)
			(can_traverse rover4 waypoint8 waypoint7)
			(can_traverse rover4 waypoint9 waypoint1)
			(channel_free general)
			(empty rover0store)
			(empty rover1store)
			(empty rover2store)
			(empty rover3store)
			(empty rover4store)
			(equipped_for_imaging rover0)
			(equipped_for_imaging rover1)
			(equipped_for_imaging rover2)
			(equipped_for_imaging rover4)
			(equipped_for_rock_analysis rover2)
			(equipped_for_rock_analysis rover4)
			(equipped_for_soil_analysis rover0)
			(equipped_for_soil_analysis rover1)
			(equipped_for_soil_analysis rover3)
			(on_board camera0 rover0)
			(on_board camera1 rover4)
			(on_board camera2 rover4)
			(on_board camera3 rover1)
			(on_board camera4 rover1)
			(on_board camera5 rover2)
			(store_of rover0store rover0)
			(store_of rover1store rover1)
			(store_of rover2store rover2)
			(store_of rover3store rover3)
			(store_of rover4store rover4)
			(supports camera0 low_res)
			(supports camera1 low_res)
			(supports camera2 low_res)
			(supports camera3 colour)
			(supports camera3 high_res)
			(supports camera4 colour)
			(supports camera4 high_res)
			(supports camera5 low_res)
			(visible waypoint0 waypoint11)
			(visible waypoint0 waypoint2)
			(visible waypoint0 waypoint5)
			(visible waypoint0 waypoint6)
			(visible waypoint1 waypoint10)
			(visible waypoint1 waypoint4)
			(visible waypoint1 waypoint5)
			(visible waypoint1 waypoint6)
			(visible waypoint1 waypoint7)
			(visible waypoint1 waypoint8)
			(visible waypoint1 waypoint9)
			(visible waypoint10 waypoint1)
			(visible waypoint10 waypoint11)
			(visible waypoint10 waypoint2)
			(visible waypoint10 waypoint4)
			(visible waypoint10 waypoint5)
			(visible waypoint10 waypoint6)
			(visible waypoint10 waypoint7)
			(visible waypoint10 waypoint8)
			(visible waypoint11 waypoint0)
			(visible waypoint11 waypoint10)
			(visible waypoint11 waypoint2)
			(visible waypoint11 waypoint5)
			(visible waypoint11 waypoint7)
			(visible waypoint11 waypoint8)
			(visible waypoint11 waypoint9)
			(visible waypoint2 waypoint0)
			(visible waypoint2 waypoint10)
			(visible waypoint2 waypoint11)
			(visible waypoint2 waypoint3)
			(visible waypoint2 waypoint5)
			(visible waypoint2 waypoint7)
			(visible waypoint3 waypoint2)
			(visible waypoint3 waypoint4)
			(visible waypoint3 waypoint6)
			(visible waypoint3 waypoint7)
			(visible waypoint3 waypoint8)
			(visible waypoint3 waypoint9)
			(visible waypoint4 waypoint1)
			(visible waypoint4 waypoint10)
			(visible waypoint4 waypoint3)
			(visible waypoint4 waypoint6)
			(visible waypoint4 waypoint7)
			(visible waypoint4 waypoint8)
			(visible waypoint4 waypoint9)
			(visible waypoint5 waypoint0)
			(visible waypoint5 waypoint1)
			(visible waypoint5 waypoint10)
			(visible waypoint5 waypoint11)
			(visible waypoint5 waypoint2)
			(visible waypoint5 waypoint7)
			(visible waypoint6 waypoint0)
			(visible waypoint6 waypoint1)
			(visible waypoint6 waypoint10)
			(visible waypoint6 waypoint3)
			(visible waypoint6 waypoint4)
			(visible waypoint6 waypoint7)
			(visible waypoint6 waypoint8)
			(visible waypoint6 waypoint9)
			(visible waypoint7 waypoint1)
			(visible waypoint7 waypoint10)
			(visible waypoint7 waypoint11)
			(visible waypoint7 waypoint2)
			(visible waypoint7 waypoint3)
			(visible waypoint7 waypoint4)
			(visible waypoint7 waypoint5)
			(visible waypoint7 waypoint6)
			(visible waypoint7 waypoint8)
			(visible waypoint8 waypoint1)
			(visible waypoint8 waypoint10)
			(visible waypoint8 waypoint11)
			(visible waypoint8 waypoint3)
			(visible waypoint8 waypoint4)
			(visible waypoint8 waypoint6)
			(visible waypoint8 waypoint7)
			(visible waypoint9 waypoint1)
			(visible waypoint9 waypoint11)
			(visible waypoint9 waypoint3)
			(visible waypoint9 waypoint4)
			(visible waypoint9 waypoint6)
			(visible_from objective0 waypoint0)
			(visible_from objective0 waypoint1)
			(visible_from objective0 waypoint2)
			(visible_from objective0 waypoint3)
			(visible_from objective1 waypoint0)
			(visible_from objective1 waypoint1)
			(visible_from objective1 waypoint2)
			(visible_from objective1 waypoint3)
			(visible_from objective1 waypoint4)
			(visible_from objective2 waypoint0)
			(visible_from objective2 waypoint1)
			(visible_from objective2 waypoint2)
			(visible_from objective2 waypoint3)
			(visible_from objective2 waypoint4)
			(visible_from objective2 waypoint5)
			(visible_from objective2 waypoint6)
			(visible_from objective2 waypoint7)
			(visible_from objective3 waypoint0)
			(visible_from objective3 waypoint1)
			(visible_from objective3 waypoint2)
			(visible_from objective3 waypoint3)
			(visible_from objective4 waypoint0)
			(visible_from objective4 waypoint1)
			(visible_from objective4 waypoint2)
			(visible_from objective4 waypoint3)
			(visible_from objective4 waypoint4)
			(visible_from objective4 waypoint5)
			(visible_from objective4 waypoint6)
			(visible_from objective4 waypoint7)
			(visible_from objective4 waypoint8)
			(visible_from objective5 waypoint0)
			(visible_from objective5 waypoint1)
			(visible_from objective5 waypoint2)
			(visible_from objective5 waypoint3)
	)
(:goal (and
(communicated_soil_data waypoint4)
(communicated_soil_data waypoint6)
(communicated_soil_data waypoint3)
(communicated_soil_data waypoint2)
(communicated_soil_data waypoint11)
(communicated_rock_data waypoint9)
(communicated_rock_data waypoint4)
(communicated_rock_data waypoint8)
(communicated_rock_data waypoint10)
(communicated_rock_data waypoint5)
(communicated_rock_data waypoint7)
(communicated_rock_data waypoint0)
(communicated_rock_data waypoint6)
(communicated_image_data objective2 low_res)
(communicated_image_data objective4 colour)
(communicated_image_data objective5 low_res)
(communicated_image_data objective2 colour)
(communicated_image_data objective4 high_res)
(communicated_image_data objective0 low_res)
)
)
)

