(define (problem grid-x6-y6-t4-k4521-l5632-p100)
(:domain grid)
(:objects
f0-0f f1-0f f2-0f f3-0f f4-0f f5-0f
f0-1f f1-1f f2-1f f3-1f f4-1f f5-1f
f0-2f f1-2f f2-2f f3-2f f4-2f f5-2f
f0-3f f1-3f f2-3f f3-3f f4-3f f5-3f
f0-4f f1-4f f2-4f f3-4f f4-4f f5-4f
f0-5f f1-5f f2-5f f3-5f f4-5f f5-5f - place
shape0 shape1 shape2 shape3 - shape
key0-0 key0-1 key0-2 key0-3
key1-0 key1-1 key1-2 key1-3 key1-4
key2-0 key2-1
key3-0 - key
)
	(:init
			(arm-empty)
			(at key0-0 f4-1f)
			(at key0-1 f1-5f)
			(at key0-2 f4-1f)
			(at key0-3 f3-0f)
			(at key1-0 f0-1f)
			(at key1-1 f5-4f)
			(at key1-2 f0-2f)
			(at key1-3 f4-3f)
			(at key1-4 f2-4f)
			(at key2-0 f0-4f)
			(at key2-1 f4-3f)
			(at key3-0 f4-1f)
			(at-robot f5-3f)
			(conn f0-0f f0-1f)
			(conn f0-0f f1-0f)
			(conn f0-1f f0-0f)
			(conn f0-1f f0-2f)
			(conn f0-1f f1-1f)
			(conn f0-2f f0-1f)
			(conn f0-2f f0-3f)
			(conn f0-2f f1-2f)
			(conn f0-3f f0-2f)
			(conn f0-3f f0-4f)
			(conn f0-3f f1-3f)
			(conn f0-4f f0-3f)
			(conn f0-4f f0-5f)
			(conn f0-4f f1-4f)
			(conn f0-5f f0-4f)
			(conn f0-5f f1-5f)
			(conn f1-0f f0-0f)
			(conn f1-0f f1-1f)
			(conn f1-0f f2-0f)
			(conn f1-1f f0-1f)
			(conn f1-1f f1-0f)
			(conn f1-1f f1-2f)
			(conn f1-1f f2-1f)
			(conn f1-2f f0-2f)
			(conn f1-2f f1-1f)
			(conn f1-2f f1-3f)
			(conn f1-2f f2-2f)
			(conn f1-3f f0-3f)
			(conn f1-3f f1-2f)
			(conn f1-3f f1-4f)
			(conn f1-3f f2-3f)
			(conn f1-4f f0-4f)
			(conn f1-4f f1-3f)
			(conn f1-4f f1-5f)
			(conn f1-4f f2-4f)
			(conn f1-5f f0-5f)
			(conn f1-5f f1-4f)
			(conn f1-5f f2-5f)
			(conn f2-0f f1-0f)
			(conn f2-0f f2-1f)
			(conn f2-0f f3-0f)
			(conn f2-1f f1-1f)
			(conn f2-1f f2-0f)
			(conn f2-1f f2-2f)
			(conn f2-1f f3-1f)
			(conn f2-2f f1-2f)
			(conn f2-2f f2-1f)
			(conn f2-2f f2-3f)
			(conn f2-2f f3-2f)
			(conn f2-3f f1-3f)
			(conn f2-3f f2-2f)
			(conn f2-3f f2-4f)
			(conn f2-3f f3-3f)
			(conn f2-4f f1-4f)
			(conn f2-4f f2-3f)
			(conn f2-4f f2-5f)
			(conn f2-4f f3-4f)
			(conn f2-5f f1-5f)
			(conn f2-5f f2-4f)
			(conn f2-5f f3-5f)
			(conn f3-0f f2-0f)
			(conn f3-0f f3-1f)
			(conn f3-0f f4-0f)
			(conn f3-1f f2-1f)
			(conn f3-1f f3-0f)
			(conn f3-1f f3-2f)
			(conn f3-1f f4-1f)
			(conn f3-2f f2-2f)
			(conn f3-2f f3-1f)
			(conn f3-2f f3-3f)
			(conn f3-2f f4-2f)
			(conn f3-3f f2-3f)
			(conn f3-3f f3-2f)
			(conn f3-3f f3-4f)
			(conn f3-3f f4-3f)
			(conn f3-4f f2-4f)
			(conn f3-4f f3-3f)
			(conn f3-4f f3-5f)
			(conn f3-4f f4-4f)
			(conn f3-5f f2-5f)
			(conn f3-5f f3-4f)
			(conn f3-5f f4-5f)
			(conn f4-0f f3-0f)
			(conn f4-0f f4-1f)
			(conn f4-0f f5-0f)
			(conn f4-1f f3-1f)
			(conn f4-1f f4-0f)
			(conn f4-1f f4-2f)
			(conn f4-1f f5-1f)
			(conn f4-2f f3-2f)
			(conn f4-2f f4-1f)
			(conn f4-2f f4-3f)
			(conn f4-2f f5-2f)
			(conn f4-3f f3-3f)
			(conn f4-3f f4-2f)
			(conn f4-3f f4-4f)
			(conn f4-3f f5-3f)
			(conn f4-4f f3-4f)
			(conn f4-4f f4-3f)
			(conn f4-4f f4-5f)
			(conn f4-4f f5-4f)
			(conn f4-5f f3-5f)
			(conn f4-5f f4-4f)
			(conn f4-5f f5-5f)
			(conn f5-0f f4-0f)
			(conn f5-0f f5-1f)
			(conn f5-1f f4-1f)
			(conn f5-1f f5-0f)
			(conn f5-1f f5-2f)
			(conn f5-2f f4-2f)
			(conn f5-2f f5-1f)
			(conn f5-2f f5-3f)
			(conn f5-3f f4-3f)
			(conn f5-3f f5-2f)
			(conn f5-3f f5-4f)
			(conn f5-4f f4-4f)
			(conn f5-4f f5-3f)
			(conn f5-4f f5-5f)
			(conn f5-5f f4-5f)
			(conn f5-5f f5-4f)
			(key-shape key0-0 shape0)
			(key-shape key0-1 shape0)
			(key-shape key0-2 shape0)
			(key-shape key0-3 shape0)
			(key-shape key1-0 shape1)
			(key-shape key1-1 shape1)
			(key-shape key1-2 shape1)
			(key-shape key1-3 shape1)
			(key-shape key1-4 shape1)
			(key-shape key2-0 shape2)
			(key-shape key2-1 shape2)
			(key-shape key3-0 shape3)
			(lock-shape f0-1f shape1)
			(lock-shape f0-3f shape2)
			(lock-shape f0-4f shape1)
			(lock-shape f1-5f shape3)
			(lock-shape f2-0f shape0)
			(lock-shape f2-1f shape0)
			(lock-shape f2-2f shape1)
			(lock-shape f2-4f shape1)
			(lock-shape f2-5f shape0)
			(lock-shape f3-0f shape3)
			(lock-shape f3-1f shape2)
			(lock-shape f3-2f shape0)
			(lock-shape f3-4f shape0)
			(lock-shape f4-0f shape1)
			(lock-shape f4-2f shape2)
			(lock-shape f5-2f shape1)
			(locked f0-1f)
			(locked f0-3f)
			(locked f0-4f)
			(locked f1-5f)
			(locked f2-0f)
			(locked f2-1f)
			(locked f2-2f)
			(locked f2-4f)
			(locked f2-5f)
			(locked f3-0f)
			(locked f3-1f)
			(locked f3-2f)
			(locked f3-4f)
			(locked f4-0f)
			(locked f4-2f)
			(locked f5-2f)
			(open f0-0f)
			(open f0-2f)
			(open f0-5f)
			(open f1-0f)
			(open f1-1f)
			(open f1-2f)
			(open f1-3f)
			(open f1-4f)
			(open f2-3f)
			(open f3-3f)
			(open f3-5f)
			(open f4-1f)
			(open f4-3f)
			(open f4-4f)
			(open f4-5f)
			(open f5-0f)
			(open f5-1f)
			(open f5-3f)
			(open f5-4f)
			(open f5-5f)
	)
(:goal
(and
(at key0-0 f4-1f)
(at key0-1 f4-3f)
(at key0-2 f0-2f)
(at key0-3 f5-0f)
(at key1-0 f1-5f)
(at key1-1 f2-4f)
(at key1-2 f2-1f)
(at key1-3 f3-1f)
(at key1-4 f2-5f)
(at key2-0 f3-4f)
(at key2-1 f4-0f)
(at key3-0 f0-4f)
)
)
)

