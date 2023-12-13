from marllib import marl
import rllib


class TestEnv(rllib.env.multi_agent_env.MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


env = marl.make_env(environment_name="mpe", map_name="simple_spread")
mappo = marl.algos.mappo(hyperparam_source="mpe")
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
fit = mappo.fit(
    env,
    model,
    stop={"timesteps_total": 1000},
    checkpoint_freq=100,
    share_policy="group",
)
