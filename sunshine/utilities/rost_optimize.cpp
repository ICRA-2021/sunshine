//
// Created by stewart on 3/11/20.
//

#include <iostream>
#include <boost/math/distributions/lognormal.hpp>
#include "sunshine/benchmark.hpp"
#include "sunshine/common/metric.hpp"

#define USE_NLOPT

#include <limbo/bayes_opt/boptimizer.hpp> // you can also include <limbo/limbo.hpp> but it will slow down the compilation
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
    BO_PARAM(int, iterations, 25);
  };

  // we use the default parameters for acqui_ucb
  struct acqui_ucb : public defaults::acqui_ucb {
  };
};

struct Eval {
  std::string const bagfile, image_topic_name, depth_topic_name, segmentation_topic_name;
  double const cell_space = 0.8;

  // number of input dimension (x.size())
  BO_PARAM(size_t, dim_in, 3);
  // number of dimensions of the result (res.size())
  BO_PARAM(size_t, dim_out, 1);

  explicit Eval(int argc, char **argv)
        : bagfile(argv[1])
        , image_topic_name(argv[2])
        , depth_topic_name(argv[3])
        , segmentation_topic_name(argv[4])
        , cell_space((argc >= 6) ? std::stod(argv[5]) : 0.8) {
  }

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
      boost::math::lognormal alpha_dist(-2.25, 2.25);
      boost::math::lognormal beta_dist(-2.5, 2.0);
      boost::math::lognormal gamma_dist(-8.0, 4.0);
//      boost::math::lognormal space_dist(0.0, 1.0);
      double const alpha = boost::math::quantile(alpha_dist, x(0));
      double const beta = boost::math::quantile(beta_dist, x(1));
      double const gamma = 0;
//      double const cell_space = .5 * std::pow(3., x(3));
      bool const use_clahe = (x(2) >= 0.5);
//      bool const use_texton = (x(5) >= 0.75);
//      bool const use_orb = (x(6) >= 0.25);
      sunshine::Parameters params{{{"alpha", alpha},
                                        {"beta", beta},
                                        {"gamma", gamma},
                                        {"K", 10},
                                        {"V", 180 /* * use_hue */ + 256 /* * use_color */ /* + 1000 * use_texton */ + 15000 /* * use_orb*/},
                                        {"use_clahe", use_clahe},
                                        {"use_texton", false},
                                        {"use_orb", true},
                                        {"cell_space", cell_space},
                                        {"cell_time", 3600.0},
                                        {"min_obs_refine_time", 300},
                                        {"num_threads", 7}}};
      std::cerr << "Alpha: " << alpha << ", Beta: " << beta << ", Gamma: " << gamma << ", Cell Space: " << cell_space;
      std::cerr << ", CLAHE: " << use_clahe;
      std::cerr << std::endl;
      double result = sunshine::benchmark(bagfile, image_topic_name, segmentation_topic_name, depth_topic_name, params, sunshine::ami<4>, 25);
      std::cerr << "Score: " << result << std::endl;
      return tools::make_vector(result);
  }
};

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval bagfile image_topic depth_cloud_topic segmentation_topic [cell_space=0.8]" << std::endl;
        return 1;
    }

    // we use the default acquisition function / model / stat / etc.
    bayes_opt::BOptimizer<Params> boptimizer;
    // run the evaluation
    boptimizer.optimize(Eval(argc, argv));
    // the best sample found
    std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
    return 0;
}
