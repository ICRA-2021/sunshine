//
// Created by stewart on 3/11/20.
//

#include <iostream>
#include <boost/math/distributions/lognormal.hpp>
#include "sunshine/benchmark.hpp"

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
    BO_PARAM(double, noise, 0.003);
  };

  struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
  };

  // we use random samples to initialize the algorithm
  struct init_randomsampling {
    BO_PARAM(int, samples, 200);
  };

  // we stop after 40 iterations
  struct stop_maxiterations {
    BO_PARAM(int, iterations, 50);
  };

  // we use the default parameters for acqui_ucb
  struct acqui_ucb : public defaults::acqui_ucb {
  };
};

struct Eval {
  std::string const bagfile, image_topic_name, depth_topic_name, segmentation_topic_name;

  // number of input dimension (x.size())
  BO_PARAM(size_t, dim_in, 4);
  // number of dimensions of the result (res.size())
  BO_PARAM(size_t, dim_out, 1);

  Eval(char **argv)
        : bagfile(argv[1])
        , image_topic_name(argv[2])
        , depth_topic_name(argv[3])
        , segmentation_topic_name(argv[4]) {
  }

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
      boost::math::lognormal alpha_dist(-2.25, 2.25);
      boost::math::lognormal beta_dist(-2.5, 2.0);
      boost::math::lognormal space_dist(-0.7, 0.6);
      double const alpha = boost::math::quantile(alpha_dist, x(0));
      double const beta = boost::math::quantile(beta_dist, x(1));
      double const gamma = pow(10.0, 3.5 * log(x(2)));
      double const cell_space = boost::math::quantile(space_dist, x(3));
      sunshine::Parameters params{{{"alpha", alpha},
                                        {"beta", beta},
                                        {"gamma", gamma},
                                        {"K", 20},
                                        {"cell_space", cell_space},
                                        {"cell_time", 0.9},
                                        {"min_obs_refine_time", 300},
                                        {"num_threads", 7}}};
      std::cout << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << ", Cell Space: " << cell_space << std::endl;
      double result = sunshine::benchmark(bagfile, image_topic_name, segmentation_topic_name, depth_topic_name, params, sunshine::nmi, 50);
      return tools::make_vector(result);
  }
};

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic depth_topic segmentation_topic" << std::endl;
        return 1;
    }

    // we use the default acquisition function / model / stat / etc.
    bayes_opt::BOptimizer<Params> boptimizer;
    // run the evaluation
    boptimizer.optimize(Eval(argv));
    // the best sample found
    std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
    return 0;
}
