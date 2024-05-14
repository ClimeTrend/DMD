# DMD
This is a repo for developing dynamic mode decomposition for noisy climate data. 

Dynamic mode decomposition is a plethora of data-driven physics-based machine learning techniques for uncovering coherent spatio-temporal structures in the data. 

![Demo](https://github.com/ClimeTrend/DMD/assets/20075514/ff0f6755-b21f-48ff-b2bb-a6293f9558bc)



DMD does not really require batching the training data in the deep-AI sense. 

Directory `/modules` contains `.py` dependencies. 

Directory `/notebooks` contains research notebooks. 

Directory `/data` contains small sparse data, used for testing. 

I (@pyatsysh) recommend to start by running `/notebooks/Demo.ipynb`. This notebook contains a minimal example of applying DMD to climate-like dataset. The actual data is generated from an advection-diffusion PDE.

Due to (1) some known shortcomings of PyDMD package, and (2) specifics of climate data, I re-implemented some aspects (e.g., time-delay, DMD prediction, and some more). This one dependecy: `BOPDMD`. The main reason for re-implementing was Uncertainty Quanitification (UQ), which I found presently does not work as expected in PyDMD. 

The following capabilities are so far implemented:

* Use boolean masks to select subsets of image. E.g., 
* Window average of training data. E.g. take window-mean over T snapshots
* Preliminary work for Uncertainty Quantification: time-delay is re-implemented
* Train DMD, using the most stable version from PyDMD package. 
* Extract Koopman eigenvalues and eigenfunctions (modes) from trained DMD
* Run DMD forward


# Data model
DMD algorithms natively work with "snapshot matrices" of shape `(N_x, N_t)`, where `N_x` is the number of pixels and `N_t` is the number of observations. In addition, the user typically provies the corresponding array of time, of shape `(N_t, )`. 

Climate data is naturally periodic, e.g. Years, Months, Days, etc. Presently, it is assumed that training data comes in the form of daily images, stored as `ndarray`s of shape `(ny, nx)`. And that daily observations are available for `N_years`. 

Thus, training data is a list of size `N_years`, where each element is `ndarray` array of shape `(N_days, ny, nx)`. A point in time is identified by its year and day within the year. Different "years" may have different numbers of days. 


# Short Term Plan:
0. Add `requirements.txt`. At the moment, dependencies are self-evident: just run `/notebooks/Demo.ipynb`;
1. Review current data model (list of ndarrays);
2. Set up data pipeline. E.g., select one scalar field for now;
2. Evaluation metrics - WeatherBench;
3. Implement a climatology model as benchmark;
4. Uncertainty Quantification - DMD with bagging (Peter)
