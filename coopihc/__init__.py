from .agents.BaseAgent import BaseAgent
from .agents.ExampleUser import ExampleUser
from .agents.ExampleAssistant import ExampleAssistant

from .agents.lqrcontrollers.FHDT_LQRController import FHDT_LQRController
from .agents.lqrcontrollers.IHCT_LQGController import IHCT_LQGController
from .agents.lqrcontrollers.IHDT_LQRController import IHDT_LQRController
from .agents.lqrcontrollers.LQRController import LQRController

from .bundle.BaseBundle import BaseBundle
from .bundle.Bundle import Bundle
from .bundle.wrappers.Train import TrainGym
from .bundle.WsServer import WsServer
from .bundle.wrappers import PipedTaskBundleWrapper


from .inference.BaseInferenceEngine import BaseInferenceEngine
from .inference.ExampleInferenceEngine import ExampleInferenceEngine
from .inference.ContinuousKalmanUpdate import ContinuousKalmanUpdate
from .inference.GoalInferenceWithUserPolicyGiven import GoalInferenceWithUserPolicyGiven
from .inference.LinearGaussianContinuous import LinearGaussianContinuous

from .interactiontask.ClassicControlTask import ClassicControlTask
from .interactiontask.InteractionTask import InteractionTask
from .interactiontask.ExampleTask import ExampleTask

from .observation.BaseObservationEngine import BaseObservationEngine
from .observation.CascadedObservationEngine import CascadedObservationEngine
from .observation.RuleObservationEngine import RuleObservationEngine
from .observation.WrapAsObservationEngine import WrapAsObservationEngine
from .observation.ExampleObservationEngine import ExampleObservationEngine

from .policy.BasePolicy import BasePolicy
from .policy.BIGDiscretePolicy import BIGDiscretePolicy
from .policy.ELLDiscretePolicy import ELLDiscretePolicy
from .policy.LinearFeedback import LinearFeedback
from .policy.RLPolicy import RLPolicy
from .policy.WrapAsPolicy import WrapAsPolicy
from .policy.ExamplePolicy import ExamplePolicy


from .base.Space import BaseSpace
from .base.Space import Numeric
from .base.Space import CatSet
from .base.Space import Space

from .base.State import State
from .base.StateElement import StateElement

# ---------------- warnings
from .base.utils import StateNotContainedWarning
from .base.utils import NotKnownSerializationWarning
from .base.utils import ContinuousSpaceIntIndexingWarning
from .base.utils import NumpyFunctionNotHandledWarning
from .base.utils import RedefiningHandledFunctionWarning
from .base.utils import WrongConvertorWarning

# ----------------- errors
from .base.utils import SpaceLengthError
from .base.utils import StateNotContainedError
from .base.utils import SpacesNotIdenticalError
from .base.utils import NotASpaceError

# -------------------- shortcuts
# from .base.elements import lin_space
from .base.elements import integer_set
from .base.elements import integer_space
from .base.elements import box_space
from .base.elements import array_element
from .base.elements import discrete_array_element
from .base.elements import cat_element


from .observation.utils import oracle_engine_specification
from .observation.utils import blind_engine_specification
from .observation.utils import base_task_engine_specification
from .observation.utils import base_user_engine_specification
from .observation.utils import base_assistant_engine_specification

# ---------------------- pointing examples
from .examples.simplepointing.envs import SimplePointingTask
from .examples.simplepointing.users import CarefulPointer
from .examples.simplepointing.assistants import ConstantCDGain
