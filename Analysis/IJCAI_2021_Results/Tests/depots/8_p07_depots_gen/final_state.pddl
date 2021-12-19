(define (problem depotprob134536825) (:domain depots)
(:objects
depot0 depot1 depot2 depot3 depot4 - depot
distributor0 distributor1 distributor2 distributor3 distributor4 - distributor
truck0 truck1 truck2 truck3 truck4 - truck
pallet0 pallet1 pallet2 pallet3 pallet4 pallet5 pallet6 pallet7 pallet8 pallet9 - pallet
crate0 crate1 crate2 crate3 crate4 crate5 - crate
hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 hoist6 hoist7 hoist8 hoist9 - hoist)
	(:init
			(at crate0 distributor2)
			(at crate1 depot3)
			(at crate2 distributor3)
			(at crate3 depot4)
			(at crate4 distributor0)
			(at crate5 distributor0)
			(at hoist0 depot0)
			(at hoist1 depot1)
			(at hoist2 depot2)
			(at hoist3 depot3)
			(at hoist4 depot4)
			(at hoist5 distributor0)
			(at hoist6 distributor1)
			(at hoist7 distributor2)
			(at hoist8 distributor3)
			(at hoist9 distributor4)
			(at pallet0 depot0)
			(at pallet1 depot1)
			(at pallet2 depot2)
			(at pallet3 depot3)
			(at pallet4 depot4)
			(at pallet5 distributor0)
			(at pallet6 distributor1)
			(at pallet7 distributor2)
			(at pallet8 distributor3)
			(at pallet9 distributor4)
			(at truck0 distributor1)
			(at truck1 distributor3)
			(at truck2 depot0)
			(at truck3 depot2)
			(at truck4 depot2)
			(available hoist0)
			(available hoist1)
			(available hoist2)
			(available hoist3)
			(available hoist4)
			(available hoist5)
			(available hoist6)
			(available hoist7)
			(available hoist8)
			(available hoist9)
			(clear crate0)
			(clear crate1)
			(clear crate2)
			(clear crate3)
			(clear crate5)
			(clear pallet0)
			(clear pallet1)
			(clear pallet2)
			(clear pallet6)
			(clear pallet9)
			(on crate0 pallet7)
			(on crate1 pallet3)
			(on crate2 pallet8)
			(on crate3 pallet4)
			(on crate4 pallet5)
			(on crate5 crate4)
	)
(:goal (and
(on crate0 pallet3)
(on crate1 pallet6)
(on crate2 crate3)
(on crate3 pallet1)
(on crate4 crate5)
(on crate5 pallet8)
)
))
