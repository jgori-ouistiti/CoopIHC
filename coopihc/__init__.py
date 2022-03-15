from .agents.BaseAgent import BaseAgent
from .agents.ExampleUser import ExampleUser
from .agents.ExampleAssistant import ExampleAssistant

from .agents.lqrcontrollers.FHDT_LQRController import FHDT_LQRController
from .agents.lqrcontrollers.IHCT_LQGController import IHCT_LQGController
from .agents.lqrcontrollers.IHDT_LQRController import IHDT_LQRController
from .agents.lqrcontrollers.LQRController import LQRController

from .bundle._Bundle import _Bundle
from .bundle.Bundle import Bundle
from .bundle.wrappers.Train import TrainGym
from .bundle.WsServer import WsServer
from .bundle.wrappers import BundleWrapper, PipedTaskBundleWrapper


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


from .space.Space import space
from .space.State import State
from .space.StateElement import StateElement

# ---------------- warnings
from .space.utils import StateNotContainedWarning
from .space.utils import NotKnownSerializationWarning
from .space.utils import ContinuousSpaceIntIndexingWarning
from .space.utils import NumpyFunctionNotHandledWarning
from .space.utils import RedefiningHandledFunctionWarning
from .space.utils import WrongConvertorWarning

# ----------------- errors
from .space.utils import SpaceLengthError
from .space.utils import StateNotContainedError
from .space.utils import SpacesNotIdenticalError
from .space.utils import NotASpaceError

# -------------------- shortcuts
from .space.utils import autospace
from .space.utils import discrete_space
from .space.utils import continuous_space
from .space.utils import multidiscrete_space
from .space.StateElement import num_element
from .space.StateElement import array_element
from .space.StateElement import cat_element

from .observation.utils import oracle_engine_specification
from .observation.utils import blind_engine_specification
from .observation.utils import base_task_engine_specification
from .observation.utils import base_user_engine_specification
from .observation.utils import base_assistant_engine_specification


# ------------------ convertors
from .space.utils import GymConvertor
from .space.utils import WrongConvertorError
from .space.utils import GymForceConvertor
