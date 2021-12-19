


(define (problem grid-x7-y7-t4-k5521-l6632-p100)
(:domain grid)
(:objects 
        f0-0f f1-0f f2-0f f3-0f f4-0f f5-0f f6-0f 
        f0-1f f1-1f f2-1f f3-1f f4-1f f5-1f f6-1f 
        f0-2f f1-2f f2-2f f3-2f f4-2f f5-2f f6-2f 
        f0-3f f1-3f f2-3f f3-3f f4-3f f5-3f f6-3f 
        f0-4f f1-4f f2-4f f3-4f f4-4f f5-4f f6-4f 
        f0-5f f1-5f f2-5f f3-5f f4-5f f5-5f f6-5f 
        f0-6f f1-6f f2-6f f3-6f f4-6f f5-6f f6-6f - place
        shape0 shape1 shape2 shape3 - shape
        key0-0 key0-1 key0-2 key0-3 key0-4 
        key1-0 key1-1 key1-2 key1-3 key1-4 
        key2-0 key2-1 
        key3-0 - key
)
(:init
(arm-empty)
(key-shape key0-0 shape0)
(key-shape key0-1 shape0)
(key-shape key0-2 shape0)
(key-shape key0-3 shape0)
(key-shape key0-4 shape0)
(key-shape key1-0 shape1)
(key-shape key1-1 shape1)
(key-shape key1-2 shape1)
(key-shape key1-3 shape1)
(key-shape key1-4 shape1)
(key-shape key2-0 shape2)
(key-shape key2-1 shape2)
(key-shape key3-0 shape3)
(conn f0-0f f1-0f)
(conn f1-0f f2-0f)
(conn f2-0f f3-0f)
(conn f3-0f f4-0f)
(conn f4-0f f5-0f)
(conn f5-0f f6-0f)
(conn f0-1f f1-1f)
(conn f1-1f f2-1f)
(conn f2-1f f3-1f)
(conn f3-1f f4-1f)
(conn f4-1f f5-1f)
(conn f5-1f f6-1f)
(conn f0-2f f1-2f)
(conn f1-2f f2-2f)
(conn f2-2f f3-2f)
(conn f3-2f f4-2f)
(conn f4-2f f5-2f)
(conn f5-2f f6-2f)
(conn f0-3f f1-3f)
(conn f1-3f f2-3f)
(conn f2-3f f3-3f)
(conn f3-3f f4-3f)
(conn f4-3f f5-3f)
(conn f5-3f f6-3f)
(conn f0-4f f1-4f)
(conn f1-4f f2-4f)
(conn f2-4f f3-4f)
(conn f3-4f f4-4f)
(conn f4-4f f5-4f)
(conn f5-4f f6-4f)
(conn f0-5f f1-5f)
(conn f1-5f f2-5f)
(conn f2-5f f3-5f)
(conn f3-5f f4-5f)
(conn f4-5f f5-5f)
(conn f5-5f f6-5f)
(conn f0-6f f1-6f)
(conn f1-6f f2-6f)
(conn f2-6f f3-6f)
(conn f3-6f f4-6f)
(conn f4-6f f5-6f)
(conn f5-6f f6-6f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f2-0f f2-1f)
(conn f3-0f f3-1f)
(conn f4-0f f4-1f)
(conn f5-0f f5-1f)
(conn f6-0f f6-1f)
(conn f0-1f f0-2f)
(conn f1-1f f1-2f)
(conn f2-1f f2-2f)
(conn f3-1f f3-2f)
(conn f4-1f f4-2f)
(conn f5-1f f5-2f)
(conn f6-1f f6-2f)
(conn f0-2f f0-3f)
(conn f1-2f f1-3f)
(conn f2-2f f2-3f)
(conn f3-2f f3-3f)
(conn f4-2f f4-3f)
(conn f5-2f f5-3f)
(conn f6-2f f6-3f)
(conn f0-3f f0-4f)
(conn f1-3f f1-4f)
(conn f2-3f f2-4f)
(conn f3-3f f3-4f)
(conn f4-3f f4-4f)
(conn f5-3f f5-4f)
(conn f6-3f f6-4f)
(conn f0-4f f0-5f)
(conn f1-4f f1-5f)
(conn f2-4f f2-5f)
(conn f3-4f f3-5f)
(conn f4-4f f4-5f)
(conn f5-4f f5-5f)
(conn f6-4f f6-5f)
(conn f0-5f f0-6f)
(conn f1-5f f1-6f)
(conn f2-5f f2-6f)
(conn f3-5f f3-6f)
(conn f4-5f f4-6f)
(conn f5-5f f5-6f)
(conn f6-5f f6-6f)
(conn f1-0f f0-0f)
(conn f2-0f f1-0f)
(conn f3-0f f2-0f)
(conn f4-0f f3-0f)
(conn f5-0f f4-0f)
(conn f6-0f f5-0f)
(conn f1-1f f0-1f)
(conn f2-1f f1-1f)
(conn f3-1f f2-1f)
(conn f4-1f f3-1f)
(conn f5-1f f4-1f)
(conn f6-1f f5-1f)
(conn f1-2f f0-2f)
(conn f2-2f f1-2f)
(conn f3-2f f2-2f)
(conn f4-2f f3-2f)
(conn f5-2f f4-2f)
(conn f6-2f f5-2f)
(conn f1-3f f0-3f)
(conn f2-3f f1-3f)
(conn f3-3f f2-3f)
(conn f4-3f f3-3f)
(conn f5-3f f4-3f)
(conn f6-3f f5-3f)
(conn f1-4f f0-4f)
(conn f2-4f f1-4f)
(conn f3-4f f2-4f)
(conn f4-4f f3-4f)
(conn f5-4f f4-4f)
(conn f6-4f f5-4f)
(conn f1-5f f0-5f)
(conn f2-5f f1-5f)
(conn f3-5f f2-5f)
(conn f4-5f f3-5f)
(conn f5-5f f4-5f)
(conn f6-5f f5-5f)
(conn f1-6f f0-6f)
(conn f2-6f f1-6f)
(conn f3-6f f2-6f)
(conn f4-6f f3-6f)
(conn f5-6f f4-6f)
(conn f6-6f f5-6f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(conn f2-1f f2-0f)
(conn f3-1f f3-0f)
(conn f4-1f f4-0f)
(conn f5-1f f5-0f)
(conn f6-1f f6-0f)
(conn f0-2f f0-1f)
(conn f1-2f f1-1f)
(conn f2-2f f2-1f)
(conn f3-2f f3-1f)
(conn f4-2f f4-1f)
(conn f5-2f f5-1f)
(conn f6-2f f6-1f)
(conn f0-3f f0-2f)
(conn f1-3f f1-2f)
(conn f2-3f f2-2f)
(conn f3-3f f3-2f)
(conn f4-3f f4-2f)
(conn f5-3f f5-2f)
(conn f6-3f f6-2f)
(conn f0-4f f0-3f)
(conn f1-4f f1-3f)
(conn f2-4f f2-3f)
(conn f3-4f f3-3f)
(conn f4-4f f4-3f)
(conn f5-4f f5-3f)
(conn f6-4f f6-3f)
(conn f0-5f f0-4f)
(conn f1-5f f1-4f)
(conn f2-5f f2-4f)
(conn f3-5f f3-4f)
(conn f4-5f f4-4f)
(conn f5-5f f5-4f)
(conn f6-5f f6-4f)
(conn f0-6f f0-5f)
(conn f1-6f f1-5f)
(conn f2-6f f2-5f)
(conn f3-6f f3-5f)
(conn f4-6f f4-5f)
(conn f5-6f f5-5f)
(conn f6-6f f6-5f)
(open f0-0f)
(open f1-0f)
(open f4-0f)
(open f5-0f)
(open f0-1f)
(open f1-1f)
(open f5-1f)
(open f6-1f)
(open f0-2f)
(open f1-2f)
(open f2-2f)
(open f3-2f)
(open f4-2f)
(open f5-2f)
(open f6-2f)
(open f2-3f)
(open f3-3f)
(open f5-3f)
(open f6-3f)
(open f0-4f)
(open f1-4f)
(open f4-4f)
(open f5-4f)
(open f1-5f)
(open f2-5f)
(open f3-5f)
(open f4-5f)
(open f5-5f)
(open f2-6f)
(open f3-6f)
(open f4-6f)
(open f5-6f)
(locked f1-3f)
(lock-shape f1-3f shape0)
(locked f3-4f)
(lock-shape f3-4f shape0)
(locked f0-5f)
(lock-shape f0-5f shape0)
(locked f6-0f)
(lock-shape f6-0f shape0)
(locked f4-1f)
(lock-shape f4-1f shape0)
(locked f4-3f)
(lock-shape f4-3f shape0)
(locked f6-4f)
(lock-shape f6-4f shape1)
(locked f3-1f)
(lock-shape f3-1f shape1)
(locked f6-6f)
(lock-shape f6-6f shape1)
(locked f2-4f)
(lock-shape f2-4f shape1)
(locked f2-0f)
(lock-shape f2-0f shape1)
(locked f0-3f)
(lock-shape f0-3f shape1)
(locked f1-6f)
(lock-shape f1-6f shape2)
(locked f3-0f)
(lock-shape f3-0f shape2)
(locked f2-1f)
(lock-shape f2-1f shape2)
(locked f0-6f)
(lock-shape f0-6f shape3)
(locked f6-5f)
(lock-shape f6-5f shape3)
(at key0-0 f0-5f)
(at key0-1 f2-1f)
(at key0-2 f0-6f)
(at key0-3 f0-0f)
(at key0-4 f2-3f)
(at key1-0 f5-2f)
(at key1-1 f4-2f)
(at key1-2 f5-3f)
(at key1-3 f0-6f)
(at key1-4 f1-0f)
(at key2-0 f3-1f)
(at key2-1 f1-1f)
(at key3-0 f0-1f)
(at-robot f4-2f)
)
(:goal
(and
(at key0-0 f1-5f)
(at key0-1 f6-3f)
(at key0-2 f6-4f)
(at key0-3 f6-4f)
(at key0-4 f1-5f)
(at key1-0 f1-0f)
(at key1-1 f4-3f)
(at key1-2 f3-2f)
(at key1-3 f4-3f)
(at key1-4 f5-2f)
(at key2-0 f3-3f)
(at key2-1 f3-3f)
(at key3-0 f1-2f)
)
)
)

