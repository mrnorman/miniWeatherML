# miniWeatherML

Welcome to miniWeatherML: A playground for learning and developing Machine Learning (ML) surrogate models and workflows. It is based on a simplified weather model simulating flows such as [supercells](https://en.wikipedia.org/wiki/Supercell) that are realistic enough to be challenging and simple enough for rapid prototyping in:
* Data generation and curation
* Machine Learning model training
* ML model deployment and analysis
* End-to-end workflows

<img src="https://mrnorman.github.io/supercell_miniWeatherML.gif"/>

## Documentation: https://github.com/mrnorman/miniWeatherML/wiki

Author: Matt Norman (Oak Ridge National Laboratory), https://mrnorman.github.io

Contributors so far:
* Matt Norman (Oak Ridge National Laboratory)
* Murali Gopalakrishnan Meena (Oak Ridge National Laboratory)

Written in portable C++, miniWeatherML runs out of the box on CPUs as well as Nvidia, AMD, and Intel GPUs.

The core infrastructure of miniWeatherML is less than 1K lines of code, and the minimal meaningful module set is comprised of less than 3K lines of code, very little of which needs to be understood in full detail in order to effectively use miniWeatherML for its intended purposes.

### Check out the **[Kessler Microphysics Example Workflow](https://github.com/mrnorman/miniWeatherML/tree/main/experiments/supercell_kessler_surrogate)** to get started

<!-- ## Some questions to explore with miniWeatherML
* How do we sample data optimally to produce physically realistic ML models?
* How do we properly assess the behavior of a ML model once it's deployed online?
* How can we identify and implement physics-based constraints within the ML model?
* What frameworks work best for managing complex end-to-end workflows?
* Which ML models work the best in which circumstances?
* What challenges do we face when scaling out data generation, curation, training, deployment, and assessment?
* Can we determine in situ when a deployed ML model is likely to behave non-physically? -->
