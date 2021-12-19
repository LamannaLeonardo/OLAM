
(define (domain satellite)
  (:requirements :strips :typing)
  (:types satellite direction instrument mode)
  (:predicates
	(on_board ?i - instrument ?s - satellite)
	(supports ?i - instrument ?m - mode)
	(pointing ?s - satellite ?d - direction)
	(power_avail ?s - satellite)
	(power_on ?i - instrument)
	(calibrated ?i - instrument)
	(have_image ?d - direction ?m - mode)
	(calibration_target ?i - instrument ?d - direction))


 (:action turn_to 
:parameters (?param_1 - satellite ?param_2 - direction ?param_3 - direction) 
:precondition (and
 (pointing ?param_1 ?param_3)
)
:effect (and (pointing ?param_1 ?param_2) (not (pointing ?param_1 ?param_3))))

 (:action switch_on 
:parameters (?param_1 - instrument ?param_2 - satellite) 
:precondition (and
 (on_board ?param_1 ?param_2) (power_avail ?param_2)
)
:effect (and (power_on ?param_1) (not (calibrated ?param_1)) (not (power_avail ?param_2))))

 (:action switch_off 
:parameters (?param_1 - instrument ?param_2 - satellite) 
:precondition (and
 (on_board ?param_1 ?param_2) (power_on ?param_1)
)
:effect (and (power_avail ?param_2) (not (power_on ?param_1))))

 (:action calibrate 
:parameters (?param_1 - satellite ?param_2 - instrument ?param_3 - direction) 
:precondition (and
 (calibration_target ?param_2 ?param_3) (on_board ?param_2 ?param_1) (pointing ?param_1 ?param_3) (power_on ?param_2)
)
:effect (and (calibrated ?param_2) ))

 (:action take_image 
:parameters (?param_1 - satellite ?param_2 - direction ?param_3 - instrument ?param_4 - mode) 
:precondition (and
 (calibrated ?param_3) (on_board ?param_3 ?param_1) (pointing ?param_1 ?param_2) (power_on ?param_3) (supports ?param_3 ?param_4)
)
:effect (and (have_image ?param_2 ?param_4) ))
)




