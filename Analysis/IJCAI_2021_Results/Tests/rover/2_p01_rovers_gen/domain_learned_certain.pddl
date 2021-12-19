
(define (domain rover)
(:requirements :strips :typing)
(:types rover waypoint store camera mode lander objective)

(:predicates (at ?x - rover ?y - waypoint)
             (at_lander ?x - lander ?y - waypoint)
             (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
	     (equipped_for_soil_analysis ?r - rover)
             (equipped_for_rock_analysis ?r - rover)
             (equipped_for_imaging ?r - rover)
             (empty ?s - store)
             (have_rock_analysis ?r - rover ?w - waypoint)
             (have_soil_analysis ?r - rover ?w - waypoint)
             (full ?s - store)
	     (calibrated ?c - camera ?r - rover)
	     (supports ?c - camera ?m - mode)
             (available ?r - rover)
             (visible ?w - waypoint ?p - waypoint)
             (have_image ?r - rover ?o - objective ?m - mode)
             (communicated_soil_data ?w - waypoint)
             (communicated_rock_data ?w - waypoint)
             (communicated_image_data ?o - objective ?m - mode)
	     (at_soil_sample ?w - waypoint)
	     (at_rock_sample ?w - waypoint)
             (visible_from ?o - objective ?w - waypoint)
	     (store_of ?s - store ?r - rover)
	     (calibration_target ?i - camera ?o - objective)
	     (on_board ?i - camera ?r - rover)
	     (channel_free ?l - lander))


(:action navigate
:parameters (?param_1 - rover ?param_2 - waypoint ?param_3 - waypoint)
:precondition (and
		(at ?param_1 ?param_2)
)
:effect (and (at ?param_1 ?param_3) (not (at ?param_1 ?param_2))))

(:action sample_soil
:parameters (?param_1 - rover ?param_2 - store ?param_3 - waypoint)
:precondition (and
		(at_soil_sample ?param_3) (empty ?param_2) (equipped_for_soil_analysis ?param_1)
)
:effect (and (full ?param_2) (have_soil_analysis ?param_1 ?param_3) (not (empty ?param_2)) (not (at_soil_sample ?param_3))))

(:action sample_rock
:parameters (?param_1 - rover ?param_2 - store ?param_3 - waypoint)
:precondition (and
		(at ?param_1 ?param_3) (at_rock_sample ?param_3) (empty ?param_2)
)
:effect (and (have_rock_analysis ?param_1 ?param_3) (full ?param_2) (not (empty ?param_2)) (not (at_rock_sample ?param_3))))

(:action drop
:parameters (?param_1 - rover ?param_2 - store)
:precondition (and
		(full ?param_2)
)
:effect (and (empty ?param_2) (not (full ?param_2))))

(:action calibrate 
:parameters (?param_1 - rover ?param_2 - camera ?param_3 - objective ?param_4 - waypoint) 
:precondition (and
		(at ?param_1 ?param_4) (calibration_target ?param_2 ?param_3) (visible_from ?param_3 ?param_4)
)
:effect (and (calibrated ?param_2 ?param_1) ))

(:action take_image 
:parameters (?param_1 - rover ?param_2 - waypoint ?param_3 - objective ?param_4 - camera ?param_5 - mode) 
:precondition (and
		(at ?param_1 ?param_2) (calibrated ?param_4 ?param_1) (supports ?param_4 ?param_5) (visible_from ?param_3 ?param_2)
)
:effect (and (have_image ?param_1 ?param_3 ?param_5) (not (calibrated ?param_4 ?param_1))))

(:action communicate_soil_data 
:parameters (?param_1 - rover ?param_2 - lander ?param_3 - waypoint ?param_4 - waypoint ?param_5 - waypoint) 
:precondition (and
		(at ?param_1 ?param_4) (at_lander ?param_2 ?param_5) (have_soil_analysis ?param_1 ?param_3)
)
:effect (and (communicated_soil_data ?param_3) ))

(:action communicate_rock_data 
:parameters (?param_1 - rover ?param_2 - lander ?param_3 - waypoint ?param_4 - waypoint ?param_5 - waypoint) 
:precondition (and
		(at ?param_1 ?param_4) (at_lander ?param_2 ?param_5) (have_rock_analysis ?param_1 ?param_3)
)
:effect (and  ))

(:action communicate_image_data 
:parameters (?param_1 - rover ?param_2 - lander ?param_3 - objective ?param_4 - mode ?param_5 - waypoint ?param_6 - waypoint) 
:precondition (and
		(at ?param_1 ?param_5) (at_lander ?param_2 ?param_6) (have_image ?param_1 ?param_3 ?param_4)
)
:effect (and (communicated_image_data ?param_3 ?param_4) ))
)




























