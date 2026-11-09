"""Complete bowl-on-plate episode: 11 calls, 9 atomic function names.

Recorded EEF poses from seed11; valid only with its saved initial scene.
These are demonstration measurements, not learned Query predictions.
For another scene/robot, replace poses with grounded, calibrated EEF poses.
"""


def run_episode(atoms):
    atoms.wait(
        position=[-0.2084646605658239, 2.3313827376747686e-12, 1.17327947572964],
        orientation=[0.7066473619524775, 0.7069957135323085, -0.020076969171691366, -0.020086866391381842],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.align(
        position=[-0.20850430650372792, -0.0014663235214009175, 1.1731966050655438],
        orientation=[0.014132023283529197, 0.9998968160069739, 0.002564857512584862, -0.00025448569661856154],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.approach(
        position=[-0.048527659119545624, 0.014161490821852087, 1.104586717749601],
        orientation=[-0.0002864994542578493, 0.999993442417908, 0.0001309331299291614, -0.0036077549473713328],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.approach(
        position=[-0.049215033162489236, 0.014239771114489784, 0.9248469875250103],
        orientation=[-0.0002074026177223965, 0.9999999703885646, 5.0343289717974426e-05, 0.00011692979550276398],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.grasp(
        position=[-0.04897110623816117, 0.013972837332070026, 0.9252007744120628],
        orientation=[-0.0002294459813131367, 0.9999999339646746, 0.00023245249322285114, 0.00015934561979757364],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.lift(
        position=[-0.049439200908133085, 0.014544405984663333, 1.1016934738590416],
        orientation=[0.00023359762779517906, 0.9999998149010643, -9.441202553579274e-05, 0.0005538197856417749],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.transport(
        position=[0.10175647621969264, -0.009831980401594017, 1.1162091454994352],
        orientation=[-0.00020696399169023444, 0.9999941329557868, 0.0003322689965455883, -0.0034030599795275966],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.place(
        position=[0.10010425084840437, -0.00969647067034782, 0.9755109084402307],
        orientation=[-0.00025559632602017995, 0.9999965077302954, 0.00020603434057902238, 0.002622355350048211],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.release(
        position=[0.10105984173740117, -0.009849009027649194, 0.9762539549897692],
        orientation=[-0.00015974481650876673, 0.9999980112219884, 0.00020317669806210176, -0.001977562360782278],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.retreat(
        position=[0.1012817565978572, -0.009282570042108468, 1.1741614889997134],
        orientation=[0.00032366623363283564, 0.9999995147431732, -0.0002514862318066433, 0.0008958282550809733],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
    atoms.wait(
        position=[0.10190783585262876, -0.00908756470728778, 1.1747790290579],
        orientation=[0.0003145541430741306, 0.9999987267275562, -0.00025498175049775204, -0.0015435618757730562],
        object='akita_black_bowl_1', target='plate_1', arm='panda', frame='world',
    )
