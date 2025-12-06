This repository provides the source code for the paper "Learning Without Time-Based Embodiment Resets in Soft-Actor Critic" published at the 4th Conference on Lifelong Learning Agents (CoLLAs 2025).

# Installation
After creating a virtual environment, install the requirements with the following commands. We used Python 3.11.5.
```bash
cd embodiment-resets-study/
pip install -r requirements.txt
pip install -e .
```

If you need to run the real-robot experiment, follow the ReLoD installation instructions at https://github.com/rlai-lab/ReLoD.

# Running experiments
Continuing SAC on continuing Hopper (with embodiment resets):
```bash
cd robot/sim/
python sac_continuous_action_original.py --r_pi_update --env_id HopperCntg-v4
```

Episodic SAC on continuing Hopper (with embodiment resets):
```bash
python sac_continuous_action_original.py --env_id HopperCntg-v4
```

Continuing SAC on continuing Reacher without embodiment resets:
```bash
python sac_continuous_action_original.py --r_pi_update --env_id ReacherNoReset-v5 --no_reset True
```

Continuing SAC using ⍺-toggle intervention on continuing Reacher without embodiment resets:
```bash
python sac_continuous_action_original.py --r_pi_update --env_id ReacherNoReset-v5 --no_reset True --adapt_te True --adapt_te_high_alpha 0.02 --scale_q True
```

Continuing SAC using ⍺-toggle intervention on Visual Ball Pull-Up:
```bash
cd robot/bouncy_ball/
python ur5_bouncy_ball.py --r_pi_update --adapt_te True --adapt_te_high_alpha 0.02 --scale_q True
```

# Citation
If you find our code useful, please cite our paper:

Farrahi, H., Mahmood, A. R. (2025). Learning without time-based embodiment resets in soft-actor critic. In *Proceedings of the 4th Conference on Lifelong Learning Agents*.
