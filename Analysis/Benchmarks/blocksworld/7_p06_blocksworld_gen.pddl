

(define (problem BW-rand-9)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9  - block)
(:init
(handempty)
(on b1 b2)
(on b2 b6)
(on b3 b9)
(on b4 b8)
(on b5 b1)
(ontable b6)
(on b7 b3)
(on b8 b5)
(on b9 b4)
(clear b7)
)
(:goal
(and
(on b2 b1)
(on b3 b9)
(on b4 b2)
(on b5 b8)
(on b6 b3)
(on b7 b5)
(on b8 b4)
(on b9 b7))
)
)


