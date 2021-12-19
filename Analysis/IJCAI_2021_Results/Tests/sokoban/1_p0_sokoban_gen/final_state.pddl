(define (problem typed-sokoban-grid4-boxes1-walls2)
(:domain typed-sokoban)
(:objects
up down left right - dir
box0 - box
f0-0f f0-1f f0-2f f0-3f
f1-0f f1-1f f1-2f f1-3f
f2-0f f2-1f f2-2f f2-3f
f3-0f f3-1f f3-2f f3-3f  - loc
)
	(:init
			(adjacent f0-0f f0-1f right)
			(adjacent f0-0f f1-0f down)
			(adjacent f0-1f f0-0f left)
			(adjacent f0-1f f0-2f right)
			(adjacent f0-1f f1-1f down)
			(adjacent f0-2f f0-1f left)
			(adjacent f0-2f f0-3f right)
			(adjacent f0-2f f1-2f down)
			(adjacent f0-3f f0-2f left)
			(adjacent f0-3f f1-3f down)
			(adjacent f1-0f f0-0f up)
			(adjacent f1-0f f1-1f right)
			(adjacent f1-0f f2-0f down)
			(adjacent f1-1f f0-1f up)
			(adjacent f1-1f f1-0f left)
			(adjacent f1-1f f1-2f right)
			(adjacent f1-1f f2-1f down)
			(adjacent f1-2f f0-2f up)
			(adjacent f1-2f f1-1f left)
			(adjacent f1-2f f1-3f right)
			(adjacent f1-2f f2-2f down)
			(adjacent f1-3f f0-3f up)
			(adjacent f1-3f f1-2f left)
			(adjacent f1-3f f2-3f down)
			(adjacent f2-0f f1-0f up)
			(adjacent f2-0f f2-1f right)
			(adjacent f2-0f f3-0f down)
			(adjacent f2-1f f1-1f up)
			(adjacent f2-1f f2-0f left)
			(adjacent f2-1f f2-2f right)
			(adjacent f2-1f f3-1f down)
			(adjacent f2-2f f1-2f up)
			(adjacent f2-2f f2-1f left)
			(adjacent f2-2f f2-3f right)
			(adjacent f2-2f f3-2f down)
			(adjacent f2-3f f1-3f up)
			(adjacent f2-3f f2-2f left)
			(adjacent f2-3f f3-3f down)
			(adjacent f3-0f f2-0f up)
			(adjacent f3-0f f3-1f right)
			(adjacent f3-1f f2-1f up)
			(adjacent f3-1f f3-0f left)
			(adjacent f3-1f f3-2f right)
			(adjacent f3-2f f2-2f up)
			(adjacent f3-2f f3-1f left)
			(adjacent f3-2f f3-3f right)
			(adjacent f3-3f f2-3f up)
			(adjacent f3-3f f3-2f left)
			(at box0 f2-1f)
			(at-robot f1-1f)
			(clear f0-0f)
			(clear f0-1f)
			(clear f0-2f)
			(clear f0-3f)
			(clear f1-0f)
			(clear f1-1f)
			(clear f1-2f)
			(clear f1-3f)
			(clear f2-0f)
			(clear f2-2f)
			(clear f2-3f)
			(clear f3-0f)
			(clear f3-3f)
	)
(:goal
(and
(at box0 f3-3f)
)
)
)


