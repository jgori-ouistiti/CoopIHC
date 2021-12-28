# Version of the coopihc package
__version__ = "0.0.6"

from .agents.BaseAgent import BaseAgent
from .agents.GoalDrivenDiscreteUser import GoalDrivenDiscreteUser
from .agents.ExampleUser import ExampleUser
from .agents.ExampleAssistant import ExampleAssistant

from .agents.lqrcontrollers.FHDT_LQRController import FHDT_LQRController
from .agents.lqrcontrollers.IHCT_LQGController import IHCT_LQGController
from .agents.lqrcontrollers.IHDT_LQRController import IHDT_LQRController
from .agents.lqrcontrollers.LQRController import LQRController

from .bundle._Bundle import _Bundle
from .bundle.Bundle import Bundle
from .bundle._PlayAssistant import PlayAssistant
from .bundle._PlayBoth import PlayBoth
from .bundle._PlayNone import PlayNone
from .bundle._PlayUser import PlayUser
from .bundle._SinglePlayUser import SinglePlayUser
from .bundle._SinglePlayUserAuto import SinglePlayUserAuto
from .bundle.wrappers.Train import Train
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


from .space.Space import Space
from .space.State import State
from .space.StateElement import StateElement
from .space.utils import StateNotContainedWarning
from .space.utils import NumpyFunctionNotHandledWarning
from .space.utils import SpaceLengthError
from .space.utils import StateNotContainedError
from .space.utils import SpacesNotIdenticalError
from .space.utils import NotASpaceError
from .space.utils import autospace
from .space.utils import WrongConvertorWarning
from .space.utils import GymConvertor
from .space.utils import WrongConvertorError
from .space.utils import GymForceConvertor
from .space.utils import discrete_space
from .space.utils import continuous_space
from .space.utils import multidiscrete_space
