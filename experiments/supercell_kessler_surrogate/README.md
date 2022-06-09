# Generating a surrogate model for Kessler microphysics

Welcome to the Kessler microphysics example for creating Neural Network surrogate models. This gives a basic end-to-end workflow (not seamless, but complete) for:
1. Creating a numerical model for experimentation
2. Exploring the data that will be emulate ([gather_statistics.cpp](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/gather_statistics.cpp) and [custom_modules/gather_micro_statistics.h](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/custom_modules/gather_micro_statistics.h))
3. Generating data for training a surrogate model ([generate_micro_data.cpp](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/generate_micro_data.cpp) and [generate_micro_surrogate_data.h](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/custom_modules/generate_micro_surrogate_data.h))
4. Curating the data
5. Training a Neural Network surrogate model
6. Deploying the surrogate model online in a simulation ([inference_ponni.cpp](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/inference_ponni.cpp) and [microphysics_kessler_ponni.h](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/custom_modules/microphysics_kessler_ponni.h))

**Even simple models are complex**: What's great about this example is that it shows that even simple surrogate models can be quite difficult to formulate in such a manner that they behave physically in a real simulation. Also, exploring different "hyperparameters" to the NN architecture can demonstrate that seemingly small changes can lead to very different behavior.

**Choices can have counter-intuitive effects**: Also, architectural choices can produce counter-intuitive results. Larger networks often give worse results. More inclusive inputs often give worse results. Your model's training, validation, and test "loss" can be near machine precision in a single norm, but when investigating your training samples individually, you'll see L-infinity norms are much larger than you probably expected. Things line up so well during training, but in deployment, you typically find behavior you didn't expect.

It's a fun place to get your feet wet and to appreciate the complexity of trying to represent even simple physics with an alternative (and typically reduced) mathematical basis.

## 1. Creating a numerical model for experimentation

The [Example supercell driver](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_example/driver.cpp) gives the basic components of creating a numerical experiment. The critical components creating each experiment are the set of modules used in the time stepping. For example, this `supercell_kessler_surrogate` experiment has the following modules:

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

// ...

dycore.time_step             (coupler,dtphys); // Move the flow forward according to the Euler equations
micro .time_step             (coupler,dtphys); // Perform phase changes for water + precipitation / falling
modules::sponge_layer        (coupler,dtphys); // Damp spurious waves to the horiz. mean at model top
column_nudger.nudge_to_column(coupler,dtphys); // Nudge slightly back toward unstable profile
                                               // so that supercell persists for all time
```

The comments give the gist of what's going on. The main difference from the `supercell_example` driver is that the `supercell_kessler_surrogate` drivers use "Kessler" microphysics rather than the significantly more comples "P3" microphysics. The goal of this experiment is to try to replace the Kessler microphysics with a Neural Network surrogate model.

## 2. Exploring the data that will be emulated

**"Good" minima**: Neural Networks can behave in amazingly complex ways. But the common training techniques, from a mathematical perspective, are surprisingly simple. Many, if not most, training is based on some form of Gradient Descent technique, meaning "if this parameter change decreases error, then go that way, but not too far or you'll actually increase the error." I've always found it surprising how well Neural Networks can perform, giving this approach, but in the end, it all comes down to the data.

**Data is critical**: I doubt any Neural Network achieves a true global minimum of the "loss" function with respect to all possible training data, but with good enough data and a careful approach training, it always finds a "good" local minimum. At its heart, it's all about the **data**. So before we seek to create a surrogate model, we should strive to understand what we're creating a surrogate of. What are the dynamics at play. How do the computations behave? Are they stencil-like, do they traverse in a direction, or do they have some other behavior?

In the experimental driver [gather_statistics.cpp](https://github.com/mrnorman/miniWeatherML/blob/main/experiments/supercell_kessler_surrogate/gather_statistics.cpp), we are exploring the data in a very simple manner: what proportion of all cells over all time steps are "active" (meaning appreciable changes in the model state variables occur)?

In the driver we have the following code that saves the model coupler's state before and after microphysics and then keeps track of how many cells are active:

```C++
core::Coupler input;
coupler.clone_into(input);
micro .time_step             ( coupler , dtphys );
// Now that micro has run, the current coupler state is the "output" of micro
// Let's use the custom module function below to determine what proportion of
//     the cells have active microphysics
statistics_gatherer.gather_micro_statistics(input,coupler,dtphys,etime);
```

## 3. Generating data for training a surrogate model

Here, we are using inputs and outputs of the Kessler microphysics subroutine to train a Neural Network surrogate model to replace that subroutine.

**Which data do you use for training?** -- The data you train a surrogate model with likely has more influence on your end result than any other aspect of your workflow. In our case, we are making a relatively simple choice. We want roughly half of the samples we create to be "active" and the other half to be inactive. This will hopefully help the surrogate model get a good feel for when to change the model's state at all. Other than this, the choice of training samples is entirely stochastic, based on a pseudo-random uniform distribution. There are certainly far more optimal strategies for generating and choosing training data, but this is a simple start.

There are multiple strategies for generating and using data. Keep in mind that the larger / more monolithic the NN's application, the fewer training samples we have, and the less generalizable the NN will be.
* We use the **entire grid at once** to make predictions. This would provide valuable context that could prove beneficial. However, the Neural Network would be very large, and we would have very few samples to use for training this very large NN.
* We could take **entire vertical columns** at a time to make predictions. This will be something that's necessary for some emulations. But in this case (this will be explained soon), we can get even smaller
* We can use **small stencils** of inputs to constrain the output of a cell. This leads to very small NNs with a very large (many billions) pool of potential training data.
* We can use **temporal sequences** of data inside cells that would lend themselves to recurrent neural network techniques. This requires more work in setting up the data sampling strategy, but it could be a beneficial approach.

**Physics informs your choices**: For sake of simplicity and a large numbers of training samples, we choose in this example to use a **small vertical stencil of inputs to constrain one cell of outputs**. Most atmospheric microphysics routines operate in the vertical direction only, meaning there is no horizontal dependence over a time step. Further, the Kessler microphysics scheme is run with a very small time step, meaning information can only propagate one cell over a time step. The only propagating information is precipitation mass falling at terminal velocity downwared through the vertical column. Therefore, to constrain a cell of outputs, we only need that cell's inputs and those of the cell above it (a stencil size of two).

**Implementation**: In our case, for a single training samples, we write to file two cells of five variables as input: temperature, dry density, water vapor density, cloud liquid density, and preciptation density. For outputs, we write to file one cell of four variables: temperature, water vapor density, cloud liquid density, and precipitation density. Note that dry density is a deterining factor in the microphysics computations, but it does not change in those computations. Therefore it is an input but not an output. The same technique as the previous section in saving the model's coupler state before and after the microphysics call is what gives the necessary information for generating training samples. The NetCDF file format is used for storing the training samples.

## Curating the data

**Raw data --> Usable data**: Raw data can rarely be used as-is for training a Neural Network, and processing that data into a usable and optimal form can be extremely tedious. Luckily in this case, the process isn't too arduous. Languages must be tokenized, images must be labeled and normalized, etc. In our case, the data needs to be aggregated, shuffled, reshaped, normalized, and split into training and testing datasets. It's hard to say where data generation ends and curation begins (if there even is a dividing line), but how we produce samples from a simulation is best viewed as a part of data curation.

**Dimensions**: Once the data is generated into NetCDF files, it has structure and order to it. Samples, for instance, follow a temporal sequence. Also, the inputs are dimensioned differently than the outputs. We have `5 x 2` inputs and `4` outputs. These need to be cast in a way that is acceptable to the Neural Network architecture we choose. If we wanted convolutions over the two cells, we cannot flatten the input dimensions. If we want dense, feed-forward networks, the inputs must be flattened.

**Normalization**: We also need to normalize the inputs and outputs. Normalizing the outputs is the more important of the two because how you express your outputs determines what your error term is (i.e., what the training process actually pays attention to). In our case, temperature is orders of magnitude larger than the moist densities (vapor, cloud, and precip). Without normalization, we'll end up with a NN that does a fantastic job with temperature and a terrible job for everything else. Output normalization determines what the training process observes and what it ignores. You'll find that the input normalization matters too, though. In our case, we normalize each variable at a time for inputs and outputs separately so that variables are in the domain `[0,1]`.

**Shuffling and splitting**: The ordering of the samples needs to be shuffled ahead of time so that the dataset can be conveniently split into training and testing datasets. Different definitions are applied to these terms in different contexts, but we denote the testing dataset as one that is never used in the training process. We use the term "validation" data to denote the proportion of training data used for validation loss during a single "epoch" of training. How much of the dataset one uses for testing has some influence on the end result of the surrogate model. More testing data makes assessing generalization more robust, but the NN may be less accurate because it has seen less data.

## Training a Neural Network surrogate model

**So many choices**: This is where the potential choices really tensor out to impossibly large numbers. Surrogate model architectures and training techniques are incredibly varied. Given how the data is generated, the Neural Network will be quite small, meaning the trainable parameters will likely be in the hundreds. For this dataset, the immediate architectural choices would likely be dense feed-forward and skip-connection networks (like ResNet).

**Notation**: We choose to describe a NN architecture as a directed graph of "layers" in which each layer is nested inside the next layer. For instance: `Input -> Dense -> ReLU -> Dense -> Loss` should be interpreted as `Loss( Dense( ReLU( Dense( Input ) ) ) )`. We eventually obtain a "loss" (an error function of the NN output compared to the training output) through a nested sequence of operations, the innermost of which is the model input itself.


<!-- which loss function you use, what batch size you use, which optimizer you use with which parameters, what (if any) regularization to use, and the list continues. -->


* **Inputs**: First, we can choose to ignore the stencil of inputs and only use the inputs of the cell in question to constrain the outputs of that cell. It might surprise you how accurate this choice will be.
* **Depth**: How many layers do we include in the model? Dense layers are a linear matrix-vector multiply with a bias vector acced (`A*x+b`). Any non-linearity must come from "activation functions" wrapped around dense layers. Therefore, more depth in theory produces more non-linearity.
  - **Skip Connections**: Deep networks have a downside that the first layers in the sequence become harder to train as the network gets deeper. This is because as gradients propagate back through the model, they are multiplied successively in each layer. Because of this, they tend to become smaller (exponential decay even?) with each successive layer. One way to avoid this is "skip connection" networks like Resnet and DenseNet, which add or concatenate previous layer outputs to later layers. Thus, when propagating gradients back thorugh the model during training, gradients aren't multiplied, making the starting layers more trainable.
* **Capacity**: Model "capacity" is in essence the amount of information the model can encapsulate. In theory, models with more trainable parameters can express more complex behavior. You should seek a capacity that matches the phenomenon you're emulating. Too much capacity can lead to over-training that generalizes poorly to out-of-sample inputs. Too little reduces accuracy. The **number of neurons per layer**, **number of layers**, and **sparsity of active neurons** contributed to model capacity.
* **Activations**: Activations inject the non-linearity Neural Networks need to be applicable to real-world phenomena, and there are a lot of creative choices to use (for a few, see the [Wikipedia page](https://en.wikipedia.org/wiki/Activation_function) [and consider donating]). I won't try to summarize behaviors here except to say that where an activation function is flat, the neuron(s) wrapping it cannot be changed (zero gradient). Activations with flat regions can be very effective, but this is something to consider.
* **Loss Function**: The "loss" (AKA "error" or "cost function") is incredibly influential in determining the resulting behavior. This is where physics-informed constraints are commonly injected into a NN model's training. For instance, penalizing lack of mass or energy conservation can be achieved here. The loss is what tells the training what to pay attention to. While Mean Squared Error is arguably the most common for regression moels, there are many choices available. 
* **Regularization**: If you're concerned about potentially over-training your model, regularization is a tool you can use to potentially improve generalization. Different types of regularization can reduce variation or sparsify trainable weights.
* **Optimizer**: The optimizer you use is also very important. Nearly all opmizers use some sort of gradient information (the gradient of the model loss function with respect to each trainable parameter). Some also use higher-order variation of the loss function as well. One thing that can vary quite a bit among optimizers, though, is the step size:
  - **Step size**: Trainable parameters are updated in the direction of negative loss function gradient (i.e., they lower the error toward zero). The step size is the magnitude of the change in that direction. Too large a change, and you can miss the local minimum and actually increase error. Too small a change, and your model will take forever to train. Feel free to experiment with optimizers and optimizer parameters and see what fits best.
* **Batch size**: Batch size also exerts a surprising amount of influence on the end result of a model. Stochastic Gradient Descent (SGD) works by first shuffling the data, then updating trainable parameters after each "batch" of samples. You should think of the errors of each individual sample in a batch as being "averaged" together before updating parameters. Thus, large batches do not respond to individual errors, but they train **much** more quickly and can be more generalizable.

## Deploying the surrogate model online in a simulation

Deploying surrogate models efficiently inside a time stepping / iteration loop in a low-level language can be challenging. There are some C++ libraries available. While they apply to a large number of model architectures, they can be complex to use. Also, it isn't always clear from the documentation what's going on internally and when / if data is transferred between host and device memory spaces. It also isn't always easy to build these libraries for different hardware architectures (for instance Nvidia, AMD, and Intel GPUs).

While it isn't necessarily intended to be a replacement for those libraries, the [Portable Online Neural Network Inferencing (PONNI)](https://github.com/mrnorman/ponni) library makes deploying trained Neural Network models easy, portable, and transparent. It adds some complexity in requiring the user to specify the model architecture, but the API for forming models and defining layers is clean. With that, though, PONNI provides:
* Convenent routines for reading in weights from HDF5 files
* Efficient inferencing of an entire network in one kernel
* Full transparency as to when data is transferred, where the inferencing is executed, and what is being parallelized in the inferencing kernels
* Portability to CPUs and Nvidia, AMD, and Intel GPUs
* A large and growing number of layers to choose from

An example of creating, validating, and inferencing a small ResNet with PONNI is below:

```C++
auto inference =
    create_inference_model( Matvec( load_h5_weights<2>(fname,"/group","dataset1") ) , 
                            Bias  ( load_h5_weights<1>(fname,"/group","dataset2") ) , 
                            Relu( 5 , 0.1 )    ,   
                            Save_State<0>( 5 ) , 
                            Matvec( load_h5_weights<2>(fname,"/group","dataset3") ) , 
                            Bias  ( load_h5_weights<1>(fname,"/group","dataset4") ) , 
                            Relu( 5 , 0.1 )    ,   
                            Binop_Add<0>( 5 )  ,
                            Save_State<0>( 5 ) , 
                            Matvec( load_h5_weights<2>(fname,"/group","dataset5") ) , 
                            Bias  ( load_h5_weights<1>(fname,"/group","dataset6") ) , 
                            Relu( 5 , 0.1 )    ,   
                            Binop_Add<0>( 5 )  ,
                            Matvec( load_h5_weights<2>(fname,"/group","dataset7") ) , 
                            Bias  ( load_h5_weights<1>(fname,"/group","dataset8") ) );
inference.validate();
inference.print_verbose();
real2d inputs("inputs",num_inputs,batch_size);
// Populate the inputs ...
auto outputs = inference.batch_parallel( inputs );
```

There are many times when a model needs to be stored in a class or namespace and resused many times. In a simple feed-forward NN case, the object can be created and later assigned as:

```C++
typedef decltype(create_inference_model(Matvec(),Bias(),Relu(),Matvec(),Bias())) MODEL;
MODEL model;

void initialize(std::string fname) {
  Matvec matvec_1( load_h5_weights<2>(fname,"/group","dataset1") );
  Bias   bias_1  ( load_h5_weights<1>(fname,"/group","dataset2") );
  Relu   relu_1  ( bias_1.get_num_outputs() , 0.1 );
  Matvec matvec_2( load_h5_weights<2>(fname,"/group","dataset3") );
  Bias   bias_2  ( load_h5_weights<1>(fname,"/group","dataset4") );

  model = create_inference_model(matvec_1, bias_1, relu_1, matvec_2, bias_2);
}
```

