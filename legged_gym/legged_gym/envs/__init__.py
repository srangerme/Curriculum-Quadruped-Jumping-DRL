# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go1.go1_config import Go1Cfg, Go1CfgPPO
from .go1.go1_upwards_config import Go1UpwardsCfg, Go1UpwardsCfgPPO
from .go2.go2_config import Go2Cfg, Go2CfgPPO
from .go2.go2_upwards_config import Go2UpwardsCfg, Go2UpwardsCfgPPO
from .solo12.solo12_v3_1_config import Solo12V31Cfg, Solo12V31CfgPPO
from .solo12.solo12_v3_1_upwards_config import Solo12V31UpwardsCfg, Solo12V31UpwardsCfgPPO
from .solo12.solo12_v3_1_forward_transition_config import Solo12V31ForwardTransitionCfg, Solo12V31ForwardTransitionCfgPPO
from .solo12.solo12_v3_1_forward_transition_2_config import Solo12V31ForwardTransition2Cfg, Solo12V31ForwardTransition2CfgPPO
from .solo12.solo12_v3_1_forward_transition_3_config import Solo12V31ForwardTransition3Cfg, Solo12V31ForwardTransition3CfgPPO
from .solo12.solo12_v3_1_forward_transition_4_config import Solo12V31ForwardTransition4Cfg, Solo12V31ForwardTransition4CfgPPO
from .solo12.solo12_v3_1_y_transition_config import Solo12V31YTransitionCfg, Solo12V31YTransitionCfgPPO
from .solo12.solo12_v3_1_y_transition_2_config import Solo12V31YTransition2Cfg, Solo12V31YTransition2CfgPPO
from .solo12.solo12_v3_1_xyyaw_transition_45_config import Solo12V31XYYawTransition45Cfg, Solo12V31XYYawTransition45CfgPPO
from .solo12.solo12_v3_1_xyyaw_go2_rand_config import Solo12V31XYYawGo2RandCfg, Solo12V31XYYawGo2RandCfgPPO
from .solo12.solo12_v3_1_xyyaw_rand_stages_config import (
    Solo12V31XYYawRandStage1Cfg, Solo12V31XYYawRandStage1CfgPPO,
    Solo12V31XYYawRandStage2ActuatorCfg, Solo12V31XYYawRandStage2ActuatorCfgPPO,
    Solo12V31XYYawRandStage2Latency30Jitter5Cfg, Solo12V31XYYawRandStage2Latency30Jitter5CfgPPO,
    Solo12V31XYYawRandStage2BridgeCfg, Solo12V31XYYawRandStage2BridgeCfgPPO,
    Solo12V31XYYawRandStage2BridgeNoAssistCfg, Solo12V31XYYawRandStage2BridgeNoAssistCfgPPO,
    Solo12V31XYYawRandStage2Latency35StableCfg, Solo12V31XYYawRandStage2Latency35StableCfgPPO,
    Solo12V31XYYawRandStage2Latency40StableCfg, Solo12V31XYYawRandStage2Latency40StableCfgPPO,
    Solo12V31XYYawRandStage2Latency40Jitter25Cfg, Solo12V31XYYawRandStage2Latency40Jitter25CfgPPO,
    Solo12V31XYYawRandStage2Cfg, Solo12V31XYYawRandStage2CfgPPO,
    Solo12V31XYYawRandStage3HasJump30Cfg, Solo12V31XYYawRandStage3HasJump30CfgPPO,
    Solo12V31XYYawRandStage3HasJump60Cfg, Solo12V31XYYawRandStage3HasJump60CfgPPO,
    Solo12V31XYYawRandStage4Cfg, Solo12V31XYYawRandStage4CfgPPO,
    Solo12V31XYYawRandStage5Push02Cfg, Solo12V31XYYawRandStage5Push02CfgPPO,
    Solo12V31XYYawRandStage5Push04Cfg, Solo12V31XYYawRandStage5Push04CfgPPO,
)
from .solo12.solo12_v3_1_xyyaw_transition_config import Solo12V31XYYawTransitionCfg, Solo12V31XYYawTransitionCfgPPO
from .solo12.solo12_v3_1_xyyaw_rand_v2_config import (
    Solo12V31XYYawRandV2S0CleanCfg, Solo12V31XYYawRandV2S0CleanCfgPPO,
    Solo12V31XYYawRandV2S0BaselineUpCfg, Solo12V31XYYawRandV2S0BaselineUpCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardCfg, Solo12V31XYYawRandV2S0BaselineForwardCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardLowLrCfg, Solo12V31XYYawRandV2S0BaselineForwardLowLrCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardFarCfg, Solo12V31XYYawRandV2S0BaselineForwardFarCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfg, Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfg, Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfgPPO,
    Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfg, Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfgPPO,
    Solo12V31XYYawRandV2S0JointEndpointsUpCfg, Solo12V31XYYawRandV2S0JointEndpointsUpCfgPPO,
    Solo12V31XYYawRandV2S0JointEndpointsForwardCfg, Solo12V31XYYawRandV2S0JointEndpointsForwardCfgPPO,
    Solo12V31XYYawRandV2S1Actuator5Cfg, Solo12V31XYYawRandV2S1Actuator5CfgPPO,
    Solo12V31XYYawRandV2S1Actuator5ForwardCfg, Solo12V31XYYawRandV2S1Actuator5ForwardCfgPPO,
    Solo12V31XYYawRandV2S1Actuator75Cfg, Solo12V31XYYawRandV2S1Actuator75CfgPPO,
    Solo12V31XYYawRandV2S1Actuator75ForwardCfg, Solo12V31XYYawRandV2S1Actuator75ForwardCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10Cfg, Solo12V31XYYawRandV2S1Actuator10CfgPPO,
    Solo12V31XYYawRandV2S1Actuator10ForwardCfg, Solo12V31XYYawRandV2S1Actuator10ForwardCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10EndpointsCfg, Solo12V31XYYawRandV2S1Actuator10EndpointsCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfg, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfg, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfg, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfgPPO,
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfgPPO,
    Solo12V31XYYawRandV2S2Latency20Cfg, Solo12V31XYYawRandV2S2Latency20CfgPPO,
    Solo12V31XYYawRandV2S2Latency30Cfg, Solo12V31XYYawRandV2S2Latency30CfgPPO,
    Solo12V31XYYawRandV2S2Latency30Jitter5Cfg, Solo12V31XYYawRandV2S2Latency30Jitter5CfgPPO,
    Solo12V31XYYawRandV2S2Latency30Jitter5UpCfg, Solo12V31XYYawRandV2S2Latency30Jitter5UpCfgPPO,
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg, Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfgPPO,
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg, Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3CfgPPO,
    Solo12V31XYYawRandV2S3InitialOriMildUpCfg, Solo12V31XYYawRandV2S3InitialOriMildUpCfgPPO,
    Solo12V31XYYawRandV2S3InitialOriMildForwardCfg, Solo12V31XYYawRandV2S3InitialOriMildForwardCfgPPO,
    Solo12V31XYYawRandV2S3InitialOriFullUpCfg, Solo12V31XYYawRandV2S3InitialOriFullUpCfgPPO,
    Solo12V31XYYawRandV2S3InitialOriFullForwardCfg, Solo12V31XYYawRandV2S3InitialOriFullForwardCfgPPO,
    Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfg, Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfgPPO,
    Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg, Solo12V31XYYawRandV2S3InitialOriFullForwardFar2CfgPPO,
    Solo12V31XYYawRandV2S4InertialFullUpCfg, Solo12V31XYYawRandV2S4InertialFullUpCfgPPO,
    Solo12V31XYYawRandV2S4InertialFullForwardCfg, Solo12V31XYYawRandV2S4InertialFullForwardCfgPPO,
    Solo12V31XYYawRandV2S4InertialFullForwardFarCfg, Solo12V31XYYawRandV2S4InertialFullForwardFarCfgPPO,
    Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg, Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfgPPO,
    Solo12V31XYYawRandV2S5HasJump15UpCfg, Solo12V31XYYawRandV2S5HasJump15UpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump15ForwardCfg, Solo12V31XYYawRandV2S5HasJump15ForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump30UpCfg, Solo12V31XYYawRandV2S5HasJump30UpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfg, Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfgPPO,
    Solo12V31XYYawRandV2S5HasJump30ForwardCfg, Solo12V31XYYawRandV2S5HasJump30ForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45UpCfg, Solo12V31XYYawRandV2S5HasJump45UpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfg, Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg, Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfg, Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg, Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO,
    Solo12V31XYYawRandV2LandingRecoveryV1Cfg, Solo12V31XYYawRandV2LandingRecoveryV1CfgPPO,
    Solo12V31XYYawRandV2LandingRecoveryV2Cfg, Solo12V31XYYawRandV2LandingRecoveryV2CfgPPO,
    Solo12V31XYYawRandV2S6SurfaceMildUpCfg, Solo12V31XYYawRandV2S6SurfaceMildUpCfgPPO,
    Solo12V31XYYawRandV2S6SurfaceMildForwardCfg, Solo12V31XYYawRandV2S6SurfaceMildForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardCfg, Solo12V31XYYawRandV2S5HasJump45ForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg, Solo12V31XYYawRandV2S5HasJump45ForwardVz350CfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardVz400Cfg, Solo12V31XYYawRandV2S5HasJump45ForwardVz400CfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg, Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfg, Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfgPPO,
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfg, Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60UpCfg, Solo12V31XYYawRandV2S5HasJump60UpCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfg, Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60ForwardCfg, Solo12V31XYYawRandV2S5HasJump60ForwardCfgPPO,
    Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfg, Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfgPPO,
    Solo12V31XYYawRandV2S3InitialStateCfg, Solo12V31XYYawRandV2S3InitialStateCfgPPO,
    Solo12V31XYYawRandV2S4CalibrationCfg, Solo12V31XYYawRandV2S4CalibrationCfgPPO,
    Solo12V31XYYawRandV2S4NoiseCfg, Solo12V31XYYawRandV2S4NoiseCfgPPO,
    Solo12V31XYYawRandV2S5InertialMildCfg, Solo12V31XYYawRandV2S5InertialMildCfgPPO,
    Solo12V31XYYawRandV2S5InertialFullCfg, Solo12V31XYYawRandV2S5InertialFullCfgPPO,
    Solo12V31XYYawRandV2S6ContactMildCfg, Solo12V31XYYawRandV2S6ContactMildCfgPPO,
    Solo12V31XYYawRandV2S6ContactFullCfg, Solo12V31XYYawRandV2S6ContactFullCfgPPO,
    Solo12V31XYYawRandV2S7HasJump15Cfg, Solo12V31XYYawRandV2S7HasJump15CfgPPO,
    Solo12V31XYYawRandV2S7HasJump30Cfg, Solo12V31XYYawRandV2S7HasJump30CfgPPO,
    Solo12V31XYYawRandV2S7HasJump60Cfg, Solo12V31XYYawRandV2S7HasJump60CfgPPO,
    Solo12V31XYYawRandV2S8Push02Cfg, Solo12V31XYYawRandV2S8Push02CfgPPO,
    Solo12V31XYYawRandV2S8Push04Cfg, Solo12V31XYYawRandV2S8Push04CfgPPO,
)
import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )

task_registry.register( "go1_upwards", LeggedRobot, Go1UpwardsCfg(), Go1UpwardsCfgPPO() )
task_registry.register( "go1_forward", LeggedRobot, Go1Cfg(), Go1CfgPPO() )
task_registry.register( "go2_upwards", LeggedRobot, Go2UpwardsCfg(), Go2UpwardsCfgPPO() )
task_registry.register( "go2_forward", LeggedRobot, Go2Cfg(), Go2CfgPPO() )
task_registry.register( "solo12_v3_1_upwards", LeggedRobot, Solo12V31UpwardsCfg(), Solo12V31UpwardsCfgPPO() )
task_registry.register( "solo12_v3_1_forward", LeggedRobot, Solo12V31Cfg(), Solo12V31CfgPPO() )
task_registry.register( "solo12_v3_1_forward_transition", LeggedRobot, Solo12V31ForwardTransitionCfg(), Solo12V31ForwardTransitionCfgPPO() )
task_registry.register( "solo12_v3_1_forward_transition_2", LeggedRobot, Solo12V31ForwardTransition2Cfg(), Solo12V31ForwardTransition2CfgPPO() )
task_registry.register( "solo12_v3_1_forward_transition_3", LeggedRobot, Solo12V31ForwardTransition3Cfg(), Solo12V31ForwardTransition3CfgPPO() )
task_registry.register( "solo12_v3_1_forward_transition_4", LeggedRobot, Solo12V31ForwardTransition4Cfg(), Solo12V31ForwardTransition4CfgPPO() )
task_registry.register( "solo12_v3_1_y_transition", LeggedRobot, Solo12V31YTransitionCfg(), Solo12V31YTransitionCfgPPO() )
task_registry.register( "solo12_v3_1_y_transition_2", LeggedRobot, Solo12V31YTransition2Cfg(), Solo12V31YTransition2CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_transition_45", LeggedRobot, Solo12V31XYYawTransition45Cfg(), Solo12V31XYYawTransition45CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_go2_rand", LeggedRobot, Solo12V31XYYawGo2RandCfg(), Solo12V31XYYawGo2RandCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage1", LeggedRobot, Solo12V31XYYawRandStage1Cfg(), Solo12V31XYYawRandStage1CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_actuator", LeggedRobot, Solo12V31XYYawRandStage2ActuatorCfg(), Solo12V31XYYawRandStage2ActuatorCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_latency30_jitter5", LeggedRobot, Solo12V31XYYawRandStage2Latency30Jitter5Cfg(), Solo12V31XYYawRandStage2Latency30Jitter5CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_bridge", LeggedRobot, Solo12V31XYYawRandStage2BridgeCfg(), Solo12V31XYYawRandStage2BridgeCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_bridge_noassist", LeggedRobot, Solo12V31XYYawRandStage2BridgeNoAssistCfg(), Solo12V31XYYawRandStage2BridgeNoAssistCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_latency35_stable", LeggedRobot, Solo12V31XYYawRandStage2Latency35StableCfg(), Solo12V31XYYawRandStage2Latency35StableCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_latency40_stable", LeggedRobot, Solo12V31XYYawRandStage2Latency40StableCfg(), Solo12V31XYYawRandStage2Latency40StableCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2_latency40_jitter25", LeggedRobot, Solo12V31XYYawRandStage2Latency40Jitter25Cfg(), Solo12V31XYYawRandStage2Latency40Jitter25CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage2", LeggedRobot, Solo12V31XYYawRandStage2Cfg(), Solo12V31XYYawRandStage2CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage3_hasjump30", LeggedRobot, Solo12V31XYYawRandStage3HasJump30Cfg(), Solo12V31XYYawRandStage3HasJump30CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage3_hasjump60", LeggedRobot, Solo12V31XYYawRandStage3HasJump60Cfg(), Solo12V31XYYawRandStage3HasJump60CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage4", LeggedRobot, Solo12V31XYYawRandStage4Cfg(), Solo12V31XYYawRandStage4CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage5_push02", LeggedRobot, Solo12V31XYYawRandStage5Push02Cfg(), Solo12V31XYYawRandStage5Push02CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_stage5_push04", LeggedRobot, Solo12V31XYYawRandStage5Push04Cfg(), Solo12V31XYYawRandStage5Push04CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_transition", LeggedRobot, Solo12V31XYYawTransitionCfg(), Solo12V31XYYawTransitionCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline", LeggedRobot, Solo12V31XYYawRandV2S0CleanCfg(), Solo12V31XYYawRandV2S0CleanCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_up", LeggedRobot, Solo12V31XYYawRandV2S0BaselineUpCfg(), Solo12V31XYYawRandV2S0BaselineUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardCfg(), Solo12V31XYYawRandV2S0BaselineForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_lr5e5", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardLowLrCfg(), Solo12V31XYYawRandV2S0BaselineForwardLowLrCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardFarCfg(), Solo12V31XYYawRandV2S0BaselineForwardFarCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far_lr5e5", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfg(), Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far2_lr5e5", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfg(), Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far_cont_lr5e5", LeggedRobot, Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfg(), Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_joint_endpoints_up", LeggedRobot, Solo12V31XYYawRandV2S0JointEndpointsUpCfg(), Solo12V31XYYawRandV2S0JointEndpointsUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s0_joint_endpoints_forward", LeggedRobot, Solo12V31XYYawRandV2S0JointEndpointsForwardCfg(), Solo12V31XYYawRandV2S0JointEndpointsForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator5_up", LeggedRobot, Solo12V31XYYawRandV2S1Actuator5Cfg(), Solo12V31XYYawRandV2S1Actuator5CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator5_forward", LeggedRobot, Solo12V31XYYawRandV2S1Actuator5ForwardCfg(), Solo12V31XYYawRandV2S1Actuator5ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator75_up", LeggedRobot, Solo12V31XYYawRandV2S1Actuator75Cfg(), Solo12V31XYYawRandV2S1Actuator75CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator75_forward", LeggedRobot, Solo12V31XYYawRandV2S1Actuator75ForwardCfg(), Solo12V31XYYawRandV2S1Actuator75ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_up", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10Cfg(), Solo12V31XYYawRandV2S1Actuator10CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_forward", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10ForwardCfg(), Solo12V31XYYawRandV2S1Actuator10ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_up", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10EndpointsCfg(), Solo12V31XYYawRandV2S1Actuator10EndpointsCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfg(), Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_stable", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfg(), Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_far_stable", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfg(), Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_far2_stable", LeggedRobot, Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg(), Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency20", LeggedRobot, Solo12V31XYYawRandV2S2Latency20Cfg(), Solo12V31XYYawRandV2S2Latency20CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency30", LeggedRobot, Solo12V31XYYawRandV2S2Latency30Cfg(), Solo12V31XYYawRandV2S2Latency30CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5", LeggedRobot, Solo12V31XYYawRandV2S2Latency30Jitter5Cfg(), Solo12V31XYYawRandV2S2Latency30Jitter5CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_up", LeggedRobot, Solo12V31XYYawRandV2S2Latency30Jitter5UpCfg(), Solo12V31XYYawRandV2S2Latency30Jitter5UpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_forward", LeggedRobot, Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg(), Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_forward_far3", LeggedRobot, Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg(), Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_mild_up", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriMildUpCfg(), Solo12V31XYYawRandV2S3InitialOriMildUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_mild_forward", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriMildForwardCfg(), Solo12V31XYYawRandV2S3InitialOriMildForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_up", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriFullUpCfg(), Solo12V31XYYawRandV2S3InitialOriFullUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriFullForwardCfg(), Solo12V31XYYawRandV2S3InitialOriFullForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward_far", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfg(), Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward_far2", LeggedRobot, Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg(), Solo12V31XYYawRandV2S3InitialOriFullForwardFar2CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_up", LeggedRobot, Solo12V31XYYawRandV2S4InertialFullUpCfg(), Solo12V31XYYawRandV2S4InertialFullUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward", LeggedRobot, Solo12V31XYYawRandV2S4InertialFullForwardCfg(), Solo12V31XYYawRandV2S4InertialFullForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward_far", LeggedRobot, Solo12V31XYYawRandV2S4InertialFullForwardFarCfg(), Solo12V31XYYawRandV2S4InertialFullForwardFarCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward_balanced", LeggedRobot, Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg(), Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump15_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump15UpCfg(), Solo12V31XYYawRandV2S5HasJump15UpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump15_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump15ForwardCfg(), Solo12V31XYYawRandV2S5HasJump15ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump30UpCfg(), Solo12V31XYYawRandV2S5HasJump30UpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_up_consolidate", LeggedRobot, Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfg(), Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump30ForwardCfg(), Solo12V31XYYawRandV2S5HasJump30ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45UpCfg(), Solo12V31XYYawRandV2S5HasJump45UpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_shortreset_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfg(), Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_shortreset_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg(), Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_shortreset_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfg(), Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_shortreset_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg(), Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_landing_recovery_v1", LeggedRobot, Solo12V31XYYawRandV2LandingRecoveryV1Cfg(), Solo12V31XYYawRandV2LandingRecoveryV1CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_landing_recovery_v2", LeggedRobot, Solo12V31XYYawRandV2LandingRecoveryV2Cfg(), Solo12V31XYYawRandV2LandingRecoveryV2CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s6_surface_mild_up", LeggedRobot, Solo12V31XYYawRandV2S6SurfaceMildUpCfg(), Solo12V31XYYawRandV2S6SurfaceMildUpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s6_surface_mild_forward", LeggedRobot, Solo12V31XYYawRandV2S6SurfaceMildForwardCfg(), Solo12V31XYYawRandV2S6SurfaceMildForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardCfg(), Solo12V31XYYawRandV2S5HasJump45ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_vz350", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg(), Solo12V31XYYawRandV2S5HasJump45ForwardVz350CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_vz400", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardVz400Cfg(), Solo12V31XYYawRandV2S5HasJump45ForwardVz400CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg(), Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners_grouped", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfg(), Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners_retain", LeggedRobot, Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfg(), Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_up", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60UpCfg(), Solo12V31XYYawRandV2S5HasJump60UpCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_up_consolidate", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfg(), Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_forward", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60ForwardCfg(), Solo12V31XYYawRandV2S5HasJump60ForwardCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_forward_consolidate", LeggedRobot, Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfg(), Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s3_initial_state", LeggedRobot, Solo12V31XYYawRandV2S3InitialStateCfg(), Solo12V31XYYawRandV2S3InitialStateCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_calibration", LeggedRobot, Solo12V31XYYawRandV2S4CalibrationCfg(), Solo12V31XYYawRandV2S4CalibrationCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s4_noise", LeggedRobot, Solo12V31XYYawRandV2S4NoiseCfg(), Solo12V31XYYawRandV2S4NoiseCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_inertial_mild", LeggedRobot, Solo12V31XYYawRandV2S5InertialMildCfg(), Solo12V31XYYawRandV2S5InertialMildCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s5_inertial_full", LeggedRobot, Solo12V31XYYawRandV2S5InertialFullCfg(), Solo12V31XYYawRandV2S5InertialFullCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s6_contact_mild", LeggedRobot, Solo12V31XYYawRandV2S6ContactMildCfg(), Solo12V31XYYawRandV2S6ContactMildCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s6_contact_full", LeggedRobot, Solo12V31XYYawRandV2S6ContactFullCfg(), Solo12V31XYYawRandV2S6ContactFullCfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s7_hasjump15", LeggedRobot, Solo12V31XYYawRandV2S7HasJump15Cfg(), Solo12V31XYYawRandV2S7HasJump15CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s7_hasjump30", LeggedRobot, Solo12V31XYYawRandV2S7HasJump30Cfg(), Solo12V31XYYawRandV2S7HasJump30CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s7_hasjump60", LeggedRobot, Solo12V31XYYawRandV2S7HasJump60Cfg(), Solo12V31XYYawRandV2S7HasJump60CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s8_push02", LeggedRobot, Solo12V31XYYawRandV2S8Push02Cfg(), Solo12V31XYYawRandV2S8Push02CfgPPO() )
task_registry.register( "solo12_v3_1_xyyaw_rand_v2_s8_push04", LeggedRobot, Solo12V31XYYawRandV2S8Push04Cfg(), Solo12V31XYYawRandV2S8Push04CfgPPO() )
