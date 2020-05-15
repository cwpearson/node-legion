#include <iostream>
#include <vector>
#include <iomanip>

// use z3 to optimize the assignment? its an integer programming thing?
#include "z3++.h"
#include "z3.h"

// https://stackoverflow.com/questions/15599030/z3-performing-matrix-operations
// https://rise4fun.com/Z3/tutorial/optimization

/* `nx` * `ny` tasks, comm for stencil `order`
 */
std::vector<double> make_stencil_weight_matrix(int64_t nx, int64_t ny,
                                               int64_t bsx, int64_t bsy,
                                               int64_t order) {
  std::vector<double> ret(nx * ny * nx * ny);

  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      int64_t srcI = i * nx + j;
      // -y nbr
      if (i > 0) {
        int64_t dstI = (i - 1) * nx + j;
        ret[srcI * ny * nx + dstI] = bsx * order;
      }
      // +y nbr
      if (i + 1 < ny) {
        int64_t dstI = (i + 1) * nx + j;
        ret[srcI * ny * nx + dstI] = bsx * order;
      }
      // -x nbr
      if (j > 0) {
        int64_t dstI = (i)*nx + (j - 1);
        ret[srcI * ny * nx + dstI] = bsy * order;
      }
      // +x nbr
      if (j + 1 < nx) {
        int64_t dstI = (i)*nx + (j + 1);
        ret[srcI * ny * nx + dstI] = bsy * order;
      }
      // -x/-y nrb
      if (i > 0 && j > 0) {
        int64_t dstI = (i-1)*nx + (j - 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
      // -x/+y nrb
      if (i +1 < ny && j > 0) {
        int64_t dstI = (i+1)*nx + (j - 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
    // +x/-y nrb
      if (i > 0 && j + 1 < nx) {
        int64_t dstI = (i-1)*nx + (j + 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }
      // +x/+y nrb
      if (i +1 < ny && j+1 < nx) {
        int64_t dstI = (i+1)*nx + (j + 1);
        ret[srcI * ny * nx + dstI] = order * order;
      }


    }
  }
  return ret;
}

/* n tasks, random communication
 */
std::vector<double> make_random_symmetric_matrix(size_t n) {
  std::vector<double> ret(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      double val = double(rand()) / RAND_MAX;
      ret[i * n + j] = val;
      ret[j * n + i] = val;
    }
  }
  return ret;
}

int main(void) {

  int numAgents = 4;
  int numTasksX = 4;
  int numTasksY = 4;
  int bsx = 10;
  int bsy = 10;
  int order = 2;

  const int numTasks = numTasksX * numTasksY;

  z3::set_param("parallel.enable", true);
  z3::config cfg;
  cfg.set("auto_config", true);
  z3::context ctx(cfg);
  ctx.set("timeout", 10 * 60 * 1000);


  // assignment function, to be solved for
  // assigned each task to an agent
  z3::expr_vector f(ctx);
  for (int i = 0; i < numTasks; ++i) {
    std::stringstream name;
    name << "f_" << i;
    f.push_back(ctx.int_const(name.str().c_str()));
  }

  // weight matrix
  std::vector<double> wv = make_random_symmetric_matrix(numTasksX * numTasksY);
  wv = make_stencil_weight_matrix(numTasksX, numTasksY, bsx, bsy, order);

  // convert to z3 consts
  z3::expr_vector w(ctx);
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      std::stringstream valStr;
      valStr << wv[i * numTasks + j];
      z3::expr v = ctx.real_val(valStr.str().c_str());
      w.push_back(v);
    }
  }

  // print weight matrix
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      int64_t num, den;
      if (Z3_get_numeral_rational_int64(ctx, w[i * numTasks + j], &num, &den)) {
        std::cout << double(num) / den << " ";
      } else {
        std::cout << w[i * numTasks + j] << " ";
      }
    }
    std::cout << "\n";
  }

  std::cerr << "distance matrix\n";
  z3::expr_vector d(ctx);
  for (int i = 0; i < numAgents; ++i) {
    for (int j = 0; j < numAgents; ++j) {
      double val;
      if (i == j) {
        val = 0;
      } else if (i < numAgents / 2 && j < numAgents / 2) {
        val = 0.6;
      } else if (i >= numAgents / 2 && j >= numAgents / 2) {
        val = 0.6;
      } else {
        val = 1;
      }
      std::stringstream valStr;
      valStr << val;
      std::cerr << i << " " << j << " " << val << "\n";
      z3::expr v = ctx.real_val(valStr.str().c_str());
      d.push_back(v);
    }
  }

  z3::optimize opt(ctx);

  // each task is mapped to an agent, so each value of f needs to be >= 0 and <
  // numAgents
  for (int i = 0; i < numTasks; ++i) {
    opt.add(f[i] >= 0 && f[i] < numAgents);
  }

  // each agent must appear at most ceil(numTasks/numAgents) times
  // each agent must appear at least floor(numTasks/numAgents) times
  // another way to say this is that each agent must appear in f between floor
  // <= n <= ceil times
  std::vector<z3::expr_vector> agentMatches(numAgents, ctx);
  for (int a = 0; a < numAgents; ++a) {
    z3::expr_vector agentMatches(ctx);
    for (int t = 0; t < numTasks; ++t) {
      agentMatches.push_back(f[t] == a);
    }
    opt.add(atmost(agentMatches, (numTasks + numAgents - 1) / numAgents));
    opt.add(atleast(agentMatches, numTasks / numAgents));
  }

  // communication volume between agents.
  // compute from task assignment and task weights
  z3::expr w_a = z3::const_array(ctx.int_sort(), ctx.int_val(0));
  std::cerr << "acc\n";
  // wa[f[i]][f[j]] += w[i][j]
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      // add up all i-j tasks
      z3::expr idx = f[i] * numAgents + f[j];
      z3::expr t = z3::select(w_a, idx);
      w_a = z3::store(w_a, idx, t + w[i * numTasks + j]);
    }
  }

  std::cerr << "partials\n";
  //   std::cerr << "AFTER LB\n";
  //   std::cerr << opt << "\n";
  z3::expr_vector partials(ctx);
  for (int i = 0; i < numAgents; ++i) {
    for (int j = 0; j < numAgents; ++j) {
      z3::expr waij = z3::select(w_a, i * numAgents + j);
      partials.push_back(waij * d[i * numAgents + j]);
    }
  }

  //   std::cerr << "partials:\n";
  //     std::cerr << partials << "\n";

  std::cerr << "costs\n";
  z3::expr maxCost = partials[0];
  for (int i = 1; i < numAgents * numAgents; ++i) {
    maxCost = max(maxCost, partials[i]);
  }
  z3::expr sumCost = sum(partials);

  // std::cerr << "cost:\n";
  // std::cerr << cost << "\n";

  // first, solve for minimum max weight, then minimum sum of communication cost
  // this weight is only right for one task per agent.
  // with multiple tasks per agent, we need to sum up the communication between
  // each pair of agents and use that for the cost.
  const z3::optimize::handle h = opt.minimize(maxCost);
  const z3::optimize::handle h2 = opt.minimize(sumCost);

//   std::cerr << opt << "\n";
  std::cerr << "go!\n";
  z3::check_result res;
  try {
    res = opt.check();
  } catch (const z3::exception &e) {
    std::cerr << "EXCEPTION:" << e.what() << "\n";
    res = z3::unsat;
  }

  if (z3::sat == res) {
    std::cout << "finished!\n";
  } else {
    std::cout << "timeout?\n";
  }

  z3::model m = opt.get_model();

  if (!m.eval(maxCost).is_numeral()) {
    std::cerr << "failed to find a solution\n";
  } else {
    int64_t num, den;
    if (Z3_get_numeral_rational_int64(ctx, m.eval(maxCost), &num, &den)) {
      std::cout << "cost: " << double(num) / den << "\n";
    } else {
      std::cout << "cost: " << m.eval(maxCost) << "\n";
    }
    if (Z3_get_numeral_rational_int64(ctx, m.eval(sumCost), &num, &den)) {
      std::cout << "sum-cost: " << double(num) / den << "\n";
    } else {
      std::cout << "sum-cost: " << m.eval(sumCost) << "\n";
    }
    
    std::cerr << "partials:\n";
    for (size_t i = 0; i < numAgents; ++i) {
        for (int j = 0; j < numAgents; ++j) {
            std::cerr << std::setw(5) << m.eval(partials[i * numAgents + j]) << " ";
        }
        std::cerr << "\n";
    }

    for (size_t i = 0; i < numTasks; ++i) {
      std::cout << "task " << i << " -> " << m.eval(f[i]) << "\n";
    }
  }
  // https://stackoverflow.com/questions/23064533/statistics-in-z3, led to
  // https://stackoverflow.com/questions/18491922/interpretation-of-z3-statistics
  // (good summary)
  // https://stackoverflow.com/questions/6841193/which-statistics-indicate-an-efficient-run-of-z3
  //   std::cerr << opt.statistics();
}