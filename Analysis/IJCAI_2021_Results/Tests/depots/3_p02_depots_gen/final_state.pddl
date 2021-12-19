(define (problem depotprob134536825) (:domain depots)
(:objects
depot0 depot1 - depot
distributor0 distributor1 - distributor
truck0 truck1 - truck
pallet0 pallet1 pallet2 pallet3 - pallet
crate0 crate1 crate2 crate3 - crate
hoist0 hoist1 hoist2 hoist3 - hoist)
	(:init
			(at crate0 depot1)
			(at crate1 depot0)
			(at crate2 distributor1)
			(at crate3 distributor0)
			(at hoist0 depot0)
			(at hoist1 depot1)
			(at hoist2 distributor0)
			(at hoist3 distributor1)
			(at pallet0 depot0)
			(at pallet1 depot1)
			(at pallet2 distributor0)
			(at pallet3 distributor1)
			(at truck0 depot1)
			(at truck1 depot0)
			(available hoist0)
			(available hoist1)
			(available hoist2)
			(available hoist3)
			(clear crate0)
			(clear crate1)
			(clear crate2)
			(clear crate3)
			(on crate0 pallet1)
			(on crate1 pallet0)
			(on crate2 pallet3)
			(on crate3 pallet2)
	)
(:goal (and
(on crate0 pallet2)
(on crate1 pallet1)
(on crate2 crate3)
(on crate3 crate1)
)
))

