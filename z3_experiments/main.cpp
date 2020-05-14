#include <iostream>
#include <vector>

// use z3 to optimize the assignment? its an integer programming thing?
#include "z3++.h"
#include "z3.h"

// https://stackoverflow.com/questions/15599030/z3-performing-matrix-operations
// https://rise4fun.com/Z3/tutorial/optimization

int main(void) {

  int numAgents = 4;
  int numTasks = 25;

  z3::context ctx;
  ctx.set("timeout", 10000);

  // assignment function, to be solved for
  // assigned each task to an agent
  z3::expr_vector f(ctx);
  for (int i = 0; i < numTasks; ++i) {
    std::stringstream name;
    name << "f_" << i;
    f.push_back(ctx.int_const(name.str().c_str()));
  }

  // weight matrix
  std::vector<double> wv(numTasks * numTasks);
  for (int i = 0; i < numTasks; ++i) {
    for (int j = i; j < numTasks; ++j) {
      double val = double(rand()) / RAND_MAX;
      wv[i * numTasks + j] = val;
      wv[j * numTasks + i] = val;
    }
  }

  // convert to z3 consts
  z3::expr_vector w(ctx);
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      double val = double(rand()) / RAND_MAX;
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

  // distance matrix is a z3 array so it can be indexed into
  z3::sort Array = ctx.array_sort(ctx.int_sort(), ctx.real_sort());
  z3::expr_vector d(ctx);
  z3::expr da = ctx.constant("d", Array);
  for (int i = 0; i < numAgents; ++i) {
    for (int j = 0; j < numAgents; ++j) {
      double val;
      if (i == j) {
        val = 0;
      } else if (i < 2 && j < 2) {
        val = 0.6;
      } else if (i >= 2 && j >= 2) {
        val = 0.6;
      } else {
        val = 0.8;
      }
      std::stringstream valStr;
      valStr << val;
      z3::expr v = ctx.real_val(valStr.str().c_str());
      da = z3::store(da, i * numAgents + j, v);
    }
  }

  z3::optimize opt(ctx);

  // each value of f needs to be >= 0 and < numAgents
  for (int i = 0; i < numTasks; ++i) {
    opt.add(f[i] >= 0 && f[i] < numAgents);
  }

  //   std::cerr << "AFTER F\n";
  //   std::cerr << opt << "\n";

  // each value of f must appear at most ceil(numTasks/numAgents) times
  // each value of f must appear at least floor(numTasks/numAgents) times
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

  //   std::cerr << "AFTER LB\n";
  //   std::cerr << opt << "\n";

  z3::expr_vector partials(ctx);
  for (int i = 0; i < numTasks; ++i) {
    for (int j = 0; j < numTasks; ++j) {
      z3::expr dij = z3::select(da, f[i] * numAgents + f[j]);
      partials.push_back(w[i * numTasks + j] * dij);
    }
  }

  //   std::cerr << "partials:\n";
  //     std::cerr << partials << "\n";

  z3::expr cost = sum(partials);

  //   std::cerr << "cost:\n";
  //   std::cerr << cost << "\n";

  std::cerr << "call opt.minimize\n";
  try {

    const z3::optimize::handle h = opt.minimize(cost);
  } catch (z3::exception &e) {
    std::cerr << e.what();
    exit(EXIT_FAILURE);
  }

  if (z3::sat == opt.check()) {
    std::cout << "finished!\n";
  } else {
    std::cout << "timeout!\n";
  }

  z3::model m = opt.get_model();

  if (!m.eval(cost).is_numeral()) {
    std::cerr << "failed to find a solution\n";
  } else {
    int64_t num, den;
    if (Z3_get_numeral_rational_int64(ctx, m.eval(cost), &num, &den)) {
      std::cout << "cost: " << double(num) / den << "\n";
    } else {
      std::cout << "cost: " << m.eval(cost) << "\n";
    }

    for (size_t i = 0; i < numTasks; ++i) {
      std::cout << "task " << i << " -> " << m.eval(f[i]) << "\n";
    }
  }
}