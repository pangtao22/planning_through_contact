import numpy as np

from pydrake.all import (
    LeafSystem,
    Meshcat,
    DrakeLcm,
    AbstractValue,
    BasicVector,
    StartMeshcat,
    DiagramBuilder,
    Simulator,
    LcmInterfaceSystem,
    LcmSubscriberSystem,
    LcmScopeSystem,
)

from drake import lcmt_iiwa_status
from optitrack import optitrack_frame_t

from qsim.parser import QuasistaticParser

from control.systems_utils import wait_for_msg

from iiwa_bimanual_setup import q_model_path
from pose_estimator_box import PoseEstimatorBase
from pose_estimator_cylinder import (
    CylinderPoseEstimator,
    is_optitrack_message_good,
)

kOptitrackChannelName = "OPTITRACK_FRAMES"
kQEstimatedChannelName = "Q_SYSTEM_ESTIMATED"
kIiwaStatusChannelName = "IIWA_STATUS"


class BimanualStateEstimator(LeafSystem):
    def __init__(self, bpe: PoseEstimatorBase):
        LeafSystem.__init__(self)
        parser = QuasistaticParser(q_model_path)
        self.q_sim = parser.make_simulator_cpp()

        self.set_name("bimanual_state_estimator")
        self.iiwa_status_input_port = self.DeclareAbstractInputPort(
            "iiwa_status", AbstractValue.Make(lcmt_iiwa_status())
        )
        self.optitrack_input_port = self.DeclareAbstractInputPort(
            "optitrack", AbstractValue.Make(optitrack_frame_t)
        )
        self.n_q = self.q_sim.get_plant().num_positions()
        self.q_estimated_output_port = self.DeclareVectorOutputPort(
            "q_estimated", BasicVector(self.n_q), self.calc_q_estimated
        )

        self.bpe = bpe

    def calc_q_estimated(self, context, output):
        status_msg = self.iiwa_status_input_port.Eval(context)
        optitrack_msg = self.optitrack_input_port.Eval(context)
        X_WB = self.bpe.calc_X_WB(optitrack_msg)

        q_a = status_msg.joint_position_measured
        q_u = np.zeros(7)
        q_u[:4] = X_WB.rotation().ToQuaternion().wxyz()
        q_u[4:] = X_WB.translation()

        q = np.zeros(self.n_q)
        q[self.q_sim.get_q_u_indices_into_q()] = q_u
        q[self.q_sim.get_q_a_indices_into_q()] = q_a

        output.SetFromVector(q)


if __name__ == "__main__":
    drake_lcm = DrakeLcm()
    initial_msg = wait_for_msg(
        channel_name=kOptitrackChannelName,
        lcm_type=optitrack_frame_t,
        is_message_good=is_optitrack_message_good,
    )

    state_estimator = BimanualStateEstimator(CylinderPoseEstimator(initial_msg))

    builder = DiagramBuilder()
    builder.AddSystem(state_estimator)
    builder.AddSystem(LcmInterfaceSystem(drake_lcm))

    iiwa_status_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS", lcm_type=lcmt_iiwa_status, lcm=drake_lcm
        )
    )
    builder.Connect(
        iiwa_status_sub.get_output_port(0),
        state_estimator.iiwa_status_input_port,
    )

    optitrack_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=kOptitrackChannelName,
            lcm_type=optitrack_frame_t,
            lcm=drake_lcm,
        )
    )
    builder.Connect(
        optitrack_sub.get_output_port(0), state_estimator.optitrack_input_port
    )

    LcmScopeSystem.AddToBuilder(
        builder=builder,
        lcm=drake_lcm,
        signal=state_estimator.q_estimated_output_port,
        channel=kQEstimatedChannelName,
        publish_period=0.005,
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    iiwa_status_msg = wait_for_msg(
        channel_name=kIiwaStatusChannelName,
        lcm_type=lcmt_iiwa_status,
        is_message_good=lambda msg: msg.num_joints == 14,
    )
    context_allegro_sub = iiwa_status_sub.GetMyContextFromRoot(
        simulator.get_context()
    )
    context_allegro_sub.SetAbstractState(0, iiwa_status_msg)

    context_optitrack_sub = optitrack_sub.GetMyContextFromRoot(
        simulator.get_context()
    )
    context_optitrack_sub.SetAbstractState(0, initial_msg)

    print("Running!")
    simulator.AdvanceTo(np.inf)
