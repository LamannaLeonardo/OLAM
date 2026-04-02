

(define (problem BW-rand-8)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8  - block)
(:init
(handempty)
(ontable b1)
(on b2 b5)
(on b3 b2)
(on b4 b7)
(ontable b5)
(on b6 b3)
(on b7 b8)
(ontable b8)
(clear b1)
(clear b4)
(clear b6)
)
(:goal
(and
(on b4 b1)
(on b5 b3)
(on b6 b7)
(on b7 b2)
(on b8 b6))
)
)


