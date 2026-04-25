# env-sample-direct-api: bench summary

Baseline = `develop` HEAD (`1bfaa24`), per-step factory + wrapper allocation.
After = same HEAD + this branch's per-env `sample_next_state` / `sample_observation`
overrides that bypass the Python wrapper subclass and go straight to the
sampling math (or native C++ kernel).

Settings: 100-particle initial belief, `_simulate_path` driven on a fresh
`BeliefNode` per repeat, 1.0 s budget, median of N trials.

## POMCPOW (sims/sec, median)

| Env                                | Before  | After   | Δ       |
| ---------------------------------- | ------: | ------: | ------: |
| Tiger                              |  7,218  |  7,827  |  +8.4%  |
| Sanity                             | 10,240  | 11,160  |  +9.0%  |
| MountainCar                        |  9,939  |  9,872  |  -0.7%  |
| CartPole                           |  9,728  |  9,827  |  +1.0%  |
| DiscreteLightDark                  |  2,330  |  2,339  |  +0.4%  |
| ContinuousLightDarkDiscreteActions |  7,680  |  8,499  | +10.7%  |
| ContinuousLightDark                |  4,175  |  4,142  |  -0.8%  |
| LaserTag                           |  2,494  |  2,674  |  +7.2%  |
| ContinuousLaserTag                 |  4,010  |  4,018  |  +0.2%  |
| Push                               |  6,527  |  6,515  |  -0.2%  |
| ContinuousPush                     |  3,162  |  3,197  |  +1.1%  |
| RockSample                         |  7,258  |  7,736  |  +6.6%  |
| SafetyAnt                          |  7,199  |  7,510  |  +4.3%  |
| Pacman                             |  5,563  |  5,362  |  -3.6%  |

## PFT-DPW (sims/sec, median)

| Env                                |  Before |  After  |    Δ     |
| ---------------------------------- | ------: | ------: | -------: |
| Tiger                              |  1,480  |  1,687  |  +14.0%  |
| Sanity                             |  4,240  |  4,961  |  +17.0%  |
| MountainCar                        |  3,863  |  4,166  |   +7.8%  |
| CartPole                           |  3,842  |  4,400  |  +14.5%  |
| DiscreteLightDark                  |  279    |  357    |  +28.1%  |
| ContinuousLightDarkDiscreteActions |  2,537  |  3,115  |  +22.8%  |
| ContinuousLightDark                |  1,933  |  2,151  |  +11.3%  |
| LaserTag                           |  193    |  235    |  +21.8%  |
| ContinuousLaserTag                 |  1,639  |  1,763  |   +7.6%  |
| Push                               |  687    |  795    |  +15.7%  |
| ContinuousPush                     |  692    |  746    |   +7.8%  |
| RockSample                         |  1,023  |  1,069  |   +4.5%  |
| SafetyAnt                          |  2,505  |  3,417  |  +36.4%  |
| Pacman                             |  2,193  |  2,261  |   +3.1%  |

## Why PFT-DPW gains more than POMCPOW

PFT-DPW does a `belief.update()` per tree expansion which iterates particles
through `pomdp.sample_next_state(state=particle, action=action)` — exactly
the path the new overrides skip. POMCPOW's hot path samples one state per
sim, so it benefits primarily from the env-side wrapper-allocation savings
inside `sample_next_step`, which is a smaller share of its total runtime.

## Behavior preservation

Every modified env carries an RNG-pinned (or, for envs with non-reproducible
native RNG, statistical) equivalence test that compares the wrapper-path
draw to the new direct-path draw under identical seeds. All env test suites
pass:

- LightDark (Continuous + Discrete + DiscreteActions): 53 + 44 = 97 tests
- LaserTag (Continuous + Discrete): 184 tests
- Push (Continuous + Discrete): 151 tests
- Tiger, Sanity, MountainCar, CartPole: 144 tests
- RockSample, SafetyAnt, Pacman: 319 tests
