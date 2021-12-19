(define   (problem parking)
  (:domain parking)
  (:objects
     car_0  car_1 - car
     curb_0 curb_1 - curb
  )
  (:init
    (= (total-cost) 0)
    (at-curb car_0)
    (at-curb-num car_0 curb_0)
    (car-clear car_0)
    (at-curb car_1)
    (at-curb-num car_1 curb_1)
    (car-clear car_1)
  )
  (:goal
    (and
      (at-curb-num car_0 curb_0)
      (at-curb-num car_1 curb_1)
    )
  )
(:metric minimize (total-cost))
)
; =========== INIT =========== 
;  curb_0: car_0 
;  curb_1: car_1 
; ========== /INIT =========== 

; =========== GOAL =========== 
;  curb_0: car_0 
;  curb_1: car_1 
; =========== /GOAL =========== 
