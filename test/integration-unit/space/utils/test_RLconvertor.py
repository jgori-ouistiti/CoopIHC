import numpy
import gym


# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface"""

#     metadata = {"render.modes": ["human"]}

#     def __init__(self, *args, **kwargs):
#         super().__init__()  # Define action and observation space
#         # They must be gym.spaces objects    # Example when using discrete actions:
#         self.action_space = gym.spaces.Discrete(1)  # Example for using image as input:
#         self.observation_space = gym.spaces.Box(
#             low=numpy.ones((2, 2)), high=numpy.ones((2, 2)), dtype=numpy.float32
#         )


# def test_gymconvertor(capsys):
#     # Discrete  no offset
#     spaces = [discrete_space([0, 1, 2, 3])]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.Discrete(4)
#     assert wrapper == None
#     assert wrapperflag == [False]

#     # Discrete with offset
#     spaces = [discrete_space([2, 3, 4])]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.Discrete(3)
#     assert isinstance(wrapper(CustomEnv), gym.ActionWrapper)
#     assert wrapperflag == [True]

#     # Discrete with unequal spacing
#     spaces = [discrete_space([0, 1, 2, 4])]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.Discrete(4)
#     assert isinstance(wrapper(CustomEnv), gym.ActionWrapper)
#     assert wrapperflag == [True]

#     # Multidiscrete Real
#     spaces = [multidiscrete_space([[0, 1], [1, 2, 3], [4, 5, 6, 8]])]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.MultiDiscrete([2, 3, 4])
#     assert isinstance(wrapper(CustomEnv), gym.ActionWrapper)
#     assert wrapperflag == [False, True, True]

#     # Multidiscrete Fake
#     spaces = [discrete_space([0, 1]), discrete_space([1, 2, 3])]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.MultiDiscrete([2, 3])
#     assert isinstance(wrapper(CustomEnv), gym.ActionWrapper)
#     assert wrapperflag == [False, True]

#     # Continuous
#     spaces = [continuous_space(-numpy.ones((2, 2)), numpy.ones((2, 2)))]
#     gc = GymConvertor()
#     gymspace, wrapper, wrapperflag = gc.get_spaces_and_wrappers(spaces, "action")
#     assert gymspace == gym.spaces.Box(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
#     assert wrapper == None
#     assert wrapperflag == [False]


if __name__ == "__main__":
    # test_gymconvertor()
    pass
