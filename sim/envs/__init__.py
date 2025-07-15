from gymnasium.envs.registration import register

from gymnasium_robotics.envs.maze import maps

register(
    id='DotSeeker-v0',
    entry_point='robot.sim.envs.dot_seeker:DotSeeker',
    max_episode_steps=10000000
)

register(
    id='ReacherNoReset-v5',
    entry_point='robot.sim.envs.reacher_no_reset:ReacherEnvNoReset',
    max_episode_steps=50
)

register(
    id='HalfCheetahCntg-v4',
    entry_point='robot.sim.envs.half_cheetah_cntg:HalfCheetahCntg',
    max_episode_steps=1000
)

register(
    id='HalfCheetahSparse-v4',
    entry_point='robot.sim.envs.half_cheetah_sparse:HalfCheetahSparse',
    max_episode_steps=1000
)

register(
    id='HopperCntg-v4',
    entry_point='robot.sim.envs.hopper_cntg:HopperCntg',
    max_episode_steps=1000
)

register(
    id='Walker2dCntg-v4',
    entry_point='robot.sim.envs.walker2d_cntg:Walker2dCntg',
    max_episode_steps=1000
)

register(
    id='AntCntg-v4',
    entry_point='robot.sim.envs.ant_cntg:AntCntg',
    max_episode_steps=1000
)

register(
    id='InvertedDoublePendulumCntg-v4',
    entry_point='robot.sim.envs.inverted_double_pendulum_cntg:InvertedDoublePendulumCntg',
    max_episode_steps=1000
)

register(
    id='InvertedDoublePendulumSparse-v4',
    entry_point='robot.sim.envs.inverted_double_pendulum_sparse:InvertedDoublePendulumSparse',
    max_episode_steps=1000
)

register(
    id='InvertedPendulumSparse-v4',
    entry_point='robot.sim.envs.inverted_pendulum_sparse:InvertedPendulumSparse',
    max_episode_steps=1000
)

register(
    id='HumanoidCntg-v4',
    entry_point='robot.sim.envs.humanoid_cntg:HumanoidCntg',
    max_episode_steps=1000
)

register(
    id='FetchReachNoReset-v4',
    entry_point='robot.sim.envs.fetch_reach_no_reset:MujocoFetchReachEnvNoReset',
    max_episode_steps=50
)

register(
    id='PointMazeNoReset_Open_Diverse_G-v3',
    entry_point='robot.sim.envs.point_maze_no_reset:PointMazeEnvNoReset',
    kwargs={
        'reward_type': 'sparse',
        "maze_map": maps.OPEN_DIVERSE_G,
    },
    max_episode_steps=300
)

