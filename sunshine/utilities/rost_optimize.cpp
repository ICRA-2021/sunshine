//
// Created by stewart on 3/11/20.
//
#include <iostream>

#define USE_NLOPT

#include <limbo/bayes_opt/boptimizer.hpp> // you can also include <limbo/limbo.hpp> but it will slow down the compilation
#include "sunshine/rost_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/depth_adapter.hpp"

using namespace limbo;

struct Params {
  struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
  };

// depending on which internal optimizer we use, we need to import different parameters
#ifdef USE_NLOPT
  struct opt_nloptnograd : public defaults::opt_nloptnograd {
  };
#elif defined(USE_LIBCMAES)
  struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
  struct opt_gridsearch : public defaults::opt_gridsearch {
  };
#endif

  // enable / disable the writing of the result files
  struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, true);
  };

  // no noise
  struct kernel : public defaults::kernel {
    BO_PARAM(double, noise, 1e-10);
  };

  struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
  };

  // we use 10 random samples to initialize the algorithm
  struct init_randomsampling {
    BO_PARAM(int, samples, 10);
  };

  // we stop after 40 iterations
  struct stop_maxiterations {
    BO_PARAM(int, iterations, 40);
  };

  // we use the default parameters for acqui_ucb
  struct acqui_ucb : public defaults::acqui_ucb {
  };
};

struct Eval {
  // number of input dimension (x.size())
  BO_PARAM(size_t, dim_in, 3);
  // number of dimensions of the result (res.size())
  BO_PARAM(size_t, dim_out, 1);

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {

  }
};

int main(int argc, char **argv) {
// we use the default acquisition function / model / stat / etc.
    bayes_opt::BOptimizer<Params> boptimizer;
    // run the evaluation
    boptimizer.optimize(Eval());
    // the best sample found
    std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
    return 0;
}