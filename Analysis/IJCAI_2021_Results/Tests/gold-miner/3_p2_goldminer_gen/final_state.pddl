(define (problem typed-bomberman-rows4-cols3)
(:domain gold-miner-typed)
(:objects
f0-0f f0-1f f0-2f
f1-0f f1-1f f1-2f
f2-0f f2-1f f2-2f
f3-0f f3-1f f3-2f  - loc
)
	(:init
			(bomb-at f2-0f)
			(clear f0-0f)
			(clear f1-0f)
			(clear f1-1f)
			(clear f1-2f)
			(clear f2-0f)
			(clear f2-1f)
			(clear f2-2f)
			(clear f3-0f)
			(clear f3-2f)
			(connected f0-0f f0-1f)
			(connected f0-0f f1-0f)
			(connected f0-1f f0-0f)
			(connected f0-1f f0-2f)
			(connected f0-1f f1-1f)
			(connected f0-2f f0-1f)
			(connected f0-2f f1-2f)
			(connected f1-0f f0-0f)
			(connected f1-0f f1-1f)
			(connected f1-0f f2-0f)
			(connected f1-1f f0-1f)
			(connected f1-1f f1-0f)
			(connected f1-1f f1-2f)
			(connected f1-1f f2-1f)
			(connected f1-2f f0-2f)
			(connected f1-2f f1-1f)
			(connected f1-2f f2-2f)
			(connected f2-0f f1-0f)
			(connected f2-0f f2-1f)
			(connected f2-0f f3-0f)
			(connected f2-1f f1-1f)
			(connected f2-1f f2-0f)
			(connected f2-1f f2-2f)
			(connected f2-1f f3-1f)
			(connected f2-2f f1-2f)
			(connected f2-2f f2-1f)
			(connected f2-2f f3-2f)
			(connected f3-0f f2-0f)
			(connected f3-0f f3-1f)
			(connected f3-1f f2-1f)
			(connected f3-1f f3-0f)
			(connected f3-1f f3-2f)
			(connected f3-2f f2-2f)
			(connected f3-2f f3-1f)
			(gold-at f2-2f)
			(hard-rock-at f0-2f)
			(holds-gold)
			(laser-at f2-2f)
			(robot-at f2-2f)
			(soft-rock-at f0-1f)
			(soft-rock-at f3-1f)
	)
(:goal
(holds-gold)
)
)























