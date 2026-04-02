

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11  - block)
(:init
(handempty)
(on b1 b6)
(on b2 b10)
(ontable b3)
(ontable b4)
(on b5 b4)
(on b6 b11)
(on b7 b1)
(ontable b8)
(on b9 b5)
(ontable b10)
(on b11 b9)
(clear b2)
(clear b3)
(clear b7)
(clear b8)
)
(:goal
(and
(on b1 b8)
(on b2 b1)
(on b3 b7)
(on b5 b2)
(on b10 b9)
(on b11 b6))
)
)


