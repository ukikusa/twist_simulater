# twist_simulater

This code is a Python implementation of the twist model ([Myung et. al](https://www.nature.com/articles/s41467-018-03507-2#change-history)) with some modifications for the [paper](https://doi.org/10.1111/nph.17925).
## Dependency

- python3: check `Pipfile` for detail
- library: check `Pipfile` for detail

## Setup

- Install Python
- Donload or git clone this repository
- Run Python script e.g. `python3 "xxxx.py"`

## Papers

- An endogenous basis for synchronisation characteristics of the circadian rhythm in proliferating *Lemna minor* plants
  Kenya Ueno, Shogo Ito, Tokitaka Oyama
  *New Phytologist*, Volume233, Issue5, March 2022, Pages 2203-2215, [https://doi.org/10.1111/nph.17925](https://doi.org/10.1111/nph.17925)

## Usage

`oscillator_2D_RK4` in `twin2D_gpu`: Used for simulation.

## Citation

    @article{ueno2022endogenous,
      title={An endogenous basis for synchronisation characteristics of the circadian rhythm in proliferating Lemna minor plants},
      author={Ueno, Kenya and Ito, Shogo and Oyama, Tokitaka},
      journal={New Phytologist},
      volume={233},
      number={5},
      pages={2203--2215},
      year={2022},
      publisher={Wiley Online Library}
    }

## References

    @article{myung2018choroid,
    title={The choroid plexus is an important circadian clock component},
    author={Myung, Jihwan and Schmal, Christoph and Hong, Sungho and Tsukizawa, Yoshiaki and Rose, Pia and Zhang, Yong and Holtzman, Michael J and De Schutter, Erik and Herzel, Hanspeter and Bordyugov, Grigory and others},
    journal={Nature communications},
    volume={9},
    number={1},
    pages={1--13},
    year={2018},
    publisher={Nature Publishing Group}
    }