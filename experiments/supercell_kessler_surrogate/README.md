# Generating a surrogate model for Kessler microphysics

Welcome to the Kessler microphysics example for creating Neural Network surrogate models. This gives a basic end-to-end workflow (not seamless, but complete) for:
1. Creating a numerical model for experimentation
2. Exploring the data that will be emulated
3. Generating data for training a surrogate model
4. Curating the data
5. Training a Neural Network surrogate model
6. Deploying the surrogate model online in a simulation

What's great about this example is that it shows that even simple surrogate models can be quite difficult to formulate in such a manner that they behave physically in a real simulation. Also, exploring different "hyperparameters" to the NN architecture can demonstrate that very small changes can lead to drastically different behavior: which inputs you include, how you normalize the inputs and outputs, which loss functions you use, how many layers and how many neurons in each layer, what batch size you use, which optimizer you use with which parameters, which non-linear "activations" you use and where, what (if any) regularization to use, and the list continues.

Architectural behavior can be very counter-intuitive. Large networks often give worse results. More inclusive inputs often give worse results. Your model's training, validation, and test "loss" can be near machine precision en masse, but when investigating your training samples individually, you'll see L-infinity and L1 norms are much larger than you expected. Things line up so well during training, but in deployment, you find behavior you didn't expect.

It's a fun place to get your feet wet and to appreciate the complexity of trying to represent even simple physics with an alternative (and typically reduced) mathematical basis.

## Creating a numerical model for experimentation

The [Example supercell driver](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_example/driver.cpp) gives the basic components of creating a numerical experiment for exploring data, generating data, and deploying a neural network. The critical components of each experiment is the set of modules used in the time stepping. For example, this experiment has the following modules:

```C++
// The column nudger nudges the column-average of the model state toward the initial column-averaged state
// This is primarily for the supercell test case to keep the the instability persistently strong
modules::ColumnNudger                     column_nudger;
// Microphysics performs water phase changess + hydrometeor production, transport, collision, and aggregation
modules::Microphysics_Kessler             micro;
// They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
modules::Dynamics_Euler_Stratified_WenoFV dycore;
// To gather statistics on how frequently microphysics is active
custom_modules::StatisticsGatherer        statistics_gatherer;

dycore.time_step             (coupler,dtphys); // Move the flow forward according to the Euler equations
micro .time_step             (coupler,dtphys); // Perform phase changes for water + precipitation / falling
modules::sponge_layer        (coupler,dtphys); // Damp spurious waves to the horiz. mean at model top
column_nudger.nudge_to_column(coupler,dtphys); // Nudge slightly back toward unstable profile
                                               // so that supercell persists for all time
```

The comments give the gist of what's going on. The main difference from the `supercell_example` driver is that the `supercell_kessler_surrogate` drivers use "Kessler" microphysics rather than the significantly more comples "P3" microphysics. The goal of this experiment is to try to replace the Kessler microphysics with a Neural Network surrogate model.

## Exploring the data that will be emulated

Neural Networks can behave in amazingly complex ways. But the common training techniques, from a mathematical perspective, are surprisingly simple. Many, if not most, training is based on a Gradient Descent technique, meaning "if it's down, go that way, but not too far or it will all blow up." I've always found it surprising how well Neural Networks can perform, giving this approach, but it all comes down to the data.

I doubt any Neural Network achieves a true global minimum of the "loss" function, but with good enough data and a careful approach training, it always finds a "good" local minimum. At its heart, it's all about the **data**.

So before we seek to create 


