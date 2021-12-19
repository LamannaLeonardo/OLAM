(define (problem depotprob134536825) (:domain depots)
(:objects
depot0 - depot
distributor0 - distributor
truck0 - truck
pallet0 pallet1 - pallet
crate0 crate1 - crate
hoist0 hoist1 - hoist)
	(:init
			(at hoist0 depot0)
			(at hoist1 distributor0)
			(at pallet0 depot0)
			(at pallet1 distributor0)
			(at truck0 distributor0)
			(available hoist0)
			(clear pallet0)
			(clear pallet1)
			(in crate1 truck0)
			(lifting hoist1 crate0)
	)
(:goal (and
(on crate0 pallet0)
(on crate1 crate0)
)
))












