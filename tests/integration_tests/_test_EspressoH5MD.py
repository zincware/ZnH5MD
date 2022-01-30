from znh5md.templates import EspressoH5MD

traj = EspressoH5MD("espresso.h5")

assert traj.box[:1].shape == (1, 3)
assert traj.charge[:1].shape == (1, 250)
assert traj.force[:1].shape == (1, 250, 3)
assert traj.id[:1].shape == (1, 250)
assert traj.image[:1].shape == (1, 250, 3)
assert traj.mass[:1].shape == (1, 250)
assert traj.position[:1].shape == (1, 250, 3)
assert traj.species[:1].shape == (1, 250)
assert traj.velocity[:1].shape == (1, 250, 3)
