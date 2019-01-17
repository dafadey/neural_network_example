#include "network.h"
#include <iostream>
#include <fstream>
#include <set>
#include <typeinfo>
#include <cmath>
#include <algorithm>
#include <omp.h>

static double range_rand(double min, double max)
{
  return (double) rand() / (double) RAND_MAX * (max - min) + min;
}

void network::make_input_layer(size_t n)
{
  for(size_t i(0); i != n; i++)
  {
    sigmoid_stupid* new_sig = new sigmoid_stupid();
    input_layer_ptrs.emplace_back(new_sig);
    new_sig->inputs.push_back(sigmoid_input());
    new_sig->inputs[0].weight = 4.0;//range_rand(-1.0, 1.0);
    new_sig->bias = -2.0;//range_rand(-1.0, 1.0);
    //new_sig->inputs[0].weight = 6.0;//range_rand(-1.0, 1.0);
    //new_sig->bias = -3.0;//range_rand(-1.0, 1.0);
  }
}

void network::bind_input_layer(double* arr)
{
  size_t count(0);
  for(auto& ss : input_layer_ptrs)
  {
    ss->inputs[0].input_sigmoid = (sigmoid_base*) &arr[count];
    count ++;
  }
}

void network::add_layer(int n)
{
  std::vector<sigmoid_smart*> new_layer;
  std::vector<sigmoid_base*> prev_layer_ptrs;
  if(output_layer_ptrs.size())
  {
    for(const auto& it : output_layer_ptrs)
      prev_layer_ptrs.push_back(it);
    output_layer_ptrs.clear();
  }
  else
  {
    for(const auto& it : input_layer_ptrs)
      prev_layer_ptrs.push_back(it);
  }
  
  for(int i(0); i !=n; i++)
  {
    sigmoid_smart* new_sig = new sigmoid_smart;
    new_sig->bias = range_rand(-1.0, 1.0);
    for(auto& prev : prev_layer_ptrs)
    {
      sigmoid_input si;
      si.weight = range_rand(-1.0, 1.0);
      si.input_sigmoid = prev;
      new_sig->inputs.emplace_back(si);
    }
    new_layer.emplace_back(new_sig);
  }
  
  for(const auto& it : new_layer)
    output_layer_ptrs.push_back(it);
}

void network::make_forward_connections()
{
  for(auto& it : output_layer_ptrs)
    it->add_outputs();
}

void network::bake()
{
  std::cout << "baking network...\n";
  make_forward_connections();
  //std::set<sigmoid_base*> _all;
  
  std::vector<sigmoid_base*> todo;
  for(auto s_ptr : input_layer_ptrs)
    todo.push_back(s_ptr);
  
  while(todo.size())
  {
    //std::set<sigmoid_base*> next_set;
    std::vector<sigmoid_base*> next_set;
    for(sigmoid_base* s_ptr : todo)
    {
      if(std::find(all.begin(), all.end(), s_ptr) == all.end())
        all.push_back(s_ptr);
        //_all.insert(s_ptr);
      for(sigmoid_base* ns_ptr : s_ptr->outputs)
      {
        if(std::find(next_set.begin(), next_set.end(), ns_ptr) == next_set.end())
          next_set.push_back(ns_ptr);
        //next_set.insert(ns_ptr);
      }
    }
    //todo = next_set;
    todo.clear();
    for(auto& s : next_set)
      todo.push_back(s);
  }
  
  //for(auto s_ptr : _all)
  //  all.push_back(s_ptr);
  
  size_t total_derivatives_size(0);
  for(auto s_ptr : all)
  {
    s_ptr->derivatives.resize(s_ptr->inputs.size() + 1/*for derivative with respect to bias*/);
    total_derivatives_size += s_ptr->derivatives.size();
  }
  derivatives.resize(total_derivatives_size);
  
  std::cout << "\tdone\n";
}

void network::forward_run()
{
  reset();
  
  std::set<sigmoid_base*> todo;
  for(auto s_ptr : input_layer_ptrs)
    todo.insert(s_ptr);
  
  while(todo.size())
  {
    std::set<sigmoid_base*> next_set;
    std::set<sigmoid_base*> pending_set;
    for(auto& s_ptr : todo)
    {
      if(!s_ptr->calculate_value())
        pending_set.insert(s_ptr);
      
      for(auto ns_ptr : s_ptr->outputs)
        next_set.insert(ns_ptr);
    }
    
    todo.clear();
    
    for(auto& s_ptr : next_set)
      todo.insert(s_ptr);
    for(auto& s_ptr : pending_set)
      todo.insert(s_ptr);
  }
  fired = true;
}

void network::reset()
{
  for(auto& s_ptr : all)
    s_ptr->invalidate();
  fired = false;
}

double network::residual_sum()
{
  if(!fired)
  {
    reset();
    forward_run();
  }
  double res(.0);
  for(size_t s_id(0); s_id != output_layer_ptrs.size(); s_id++)
  {
    const auto s_ptr(output_layer_ptrs[s_id]);
    const double r(s_ptr->value - result[s_id]);
    res += r * r;
  }
  return res;
}

void network::prepare_derivative()
{
  for(sigmoid_base* s_ptr : all)
    s_ptr->invalidate_derivative();
  //std::cout << "invalidated derivative\n";
  std::set<sigmoid_base*> todo;
  
  for(size_t s_id(0); s_id != output_layer_ptrs.size(); s_id++)
  {
    const auto s_ptr(output_layer_ptrs[s_id]);
    const double r(s_ptr->value - result[s_id]);
    s_ptr->derivative_factor = 2.0 * r;
    todo.insert(s_ptr);
  }
  
  //std::cout << "first level of derivative is ready. todo size is " << todo.size() << '\n';
  
  std::set<sigmoid_base*> done;
  
  while(todo.size())
  {
    //std::cout << "doing one level of derivative preparation\n";
    std::set<sigmoid_base*> next_set;
    for(sigmoid_base* s_ptr : todo)
    {
      //std::cout << "back_fires_count=" << s_ptr->back_fires_count << ", outputs.size=" << s_ptr->outputs.size() << '\n';
      if(s_ptr->back_fires_count >= s_ptr->outputs.size())
      {
        if(done.find(s_ptr) != done.end())
          std::cout << "error 101\n";
        done.insert(s_ptr);
        double a = s_ptr->derivative_factor;
        if(!s_ptr->calculate_derivative_value())
          std::cerr << "error calcualting derivative, sigmoid does not have a value\n";
        s_ptr->derivative_factor *= a;
      }
      
      if(typeid(*s_ptr) == typeid(sigmoid_stupid))
        continue;
      
      for(auto& i : s_ptr->inputs)
      {
        i.input_sigmoid->derivative_factor += i.weight * s_ptr->derivative_factor; // accumulate derivative factors
        i.input_sigmoid->back_fires_count ++;
        next_set.insert(i.input_sigmoid);
      }
    }
    
    todo.clear();
    
    for(sigmoid_base* s_ptr : next_set)
      todo.insert(s_ptr);
  }
  
  int e102(0);
  for(auto& s_ptr : all)
  {
    if(done.find(s_ptr) == done.end())// && typeid(*s_ptr) == typeid(sigmoid_smart))
      e102++;
  }
  if(e102)
    std::cout << "error 102, count=" << e102 << '\n';
}

void network::calculate_all_derivatives()
{
  reset();
  forward_run();
  prepare_derivative();

  for(auto& s_ptr : all)
  {
    size_t i(0);
    for(; i != s_ptr->inputs.size(); i++)
    {
      if(typeid(*s_ptr) == typeid(sigmoid_stupid))
      {
        if(i != 0)
          std::cerr << "error 103 i=" << i << ", s_ptr->inputs.size()=" << s_ptr->inputs.size() << '\n';
        s_ptr->derivatives[i] = s_ptr->derivative_factor * *((double*) s_ptr->inputs[i].input_sigmoid);
      }
      else
        s_ptr->derivatives[i] = s_ptr->derivative_factor * s_ptr->inputs[i].input_sigmoid->value;
    }
    s_ptr->derivatives[i] = s_ptr->derivative_factor; // derivative with respect to bias
  }
}

void network::get_derivatives_as_vector()
{
  size_t count(0);
  for(auto& s_ptr : all)
  {
    size_t i(0);
    for(; i != s_ptr->inputs.size(); i++)
    {
      if(typeid(*s_ptr) == typeid(sigmoid_smart))
        derivatives[count] = std::make_pair(&(s_ptr->inputs[i].weight), s_ptr->derivatives[i]);
      else
        derivatives[count] = std::make_pair(&(s_ptr->inputs[i].weight), .0);
      count ++;
    }
    if(typeid(*s_ptr) == typeid(sigmoid_smart))
      derivatives[count] = std::make_pair(&(s_ptr->bias), s_ptr->derivatives[i]);
    else
      derivatives[count] = std::make_pair(&(s_ptr->bias), .0);
    count ++;
  }
}

void network::descent_single_direction(std::vector<std::pair<double* /*input images*/, double* /*ouput answers array*/>>& testing_set, std::vector<network*>& networks, double INITIAL_STEP)
{
  #define MAX_STEP_COUNT   1
  #define TARGET_STEP      .000001
  //#define INITIAL_STEP     3.0
  
  int Nthreads = networks.size();
  
  std::vector<std::vector<double>> derivatives_sum(Nthreads);
  std::vector<double> rsums(Nthreads, .0);
  
  
  #pragma omp parallel for
  for(int ni=0; ni < Nthreads; ni++)
  {
		double tss_sz_1 = 1.0 / (double) testing_set.size();
    std::vector<double>& derivative_sum = derivatives_sum[ni];
    derivative_sum.resize(derivatives.size());
    
    for(int i(0); i != derivative_sum.size(); i++)
      derivative_sum[i] = .0;

		#ifdef ACCURATESUM
		std::vector<summator> dsum(derivatives.size());
		#endif
    for(int ts_id = 0; ts_id != testing_set.size(); ts_id++)
    {
      if(ts_id % Nthreads != ni)
        continue;
      auto& ts = testing_set[ts_id]; 
      
      networks[ni]->bind_input_layer(ts.first);
      networks[ni]->result = ts.second;

      networks[ni]->calculate_all_derivatives();
      networks[ni]->get_derivatives_as_vector();
      for(size_t i(0); i != derivatives.size(); i++)
      {
				const double v = networks[ni]->derivatives[i].second * tss_sz_1;
				#ifdef ACCURATESUM
				dsum[i].MADD(v);
				#else
        derivative_sum[i] += v;
				#endif
			}
			
    }
		#ifdef ACCURATESUM
		for(size_t i(0); i != derivatives.size(); i++)
			derivative_sum[i] += dsum[i].MR();
		#endif
    
    double& rsum = rsums[ni];
    for(int ts_id = 0; ts_id != testing_set.size(); ts_id++)
    {
      if(ts_id % Nthreads != ni)
        continue;
      auto& ts = testing_set[ts_id]; 

      networks[ni]->bind_input_layer(ts.first);
      networks[ni]->result = ts.second;
      networks[ni]->reset();
      rsums[ni] += networks[ni]->residual_sum();
    }
  }

  std::vector<double> derivative_sum(derivatives.size(), .0);
  double rsum(.0);
  
  //single thread work
  //collect all partial sums to calculated partial derivatives for whole minibatch
  #ifdef ACCURATESUM
  std::vector<summator> dsum(derivative_sum.size());
  #endif
  for(int nth(0); nth<Nthreads; nth++)
  {
    for(int i=0; i!=derivative_sum.size(); i++)
    {
			const double v = derivatives_sum[nth][i];
			#ifdef ACCURATESUM
			dsum[i].MADD(v);
			#else
      derivative_sum[i] += v;
			#endif
    }
    rsum += rsums[nth];
  }
	#ifdef ACCURATESUM
	for(int i=0; i!=derivative_sum.size(); i++)
		derivative_sum[i] += dsum[i].MR();
	#endif
  
  
  double step(INITIAL_STEP);
  size_t step_count(0);
  
  while(std::abs(step) > TARGET_STEP && step_count < MAX_STEP_COUNT)
  {
    double prsum = rsum;
    
    #pragma omp parallel for
    for(int ni=0; ni < Nthreads; ni++)
    {
      for(size_t d_id(0); d_id != derivatives.size(); d_id++)
        *(networks[ni]->derivatives[d_id].first) -= derivative_sum[d_id] * step;
    }
    
    rsum = .0;
    std::vector<double> rsums(Nthreads, .0);
    #pragma omp parallel for
    for(int ni=0; ni < Nthreads; ni++)
    {
      for(int ts_id = 0; ts_id != testing_set.size(); ts_id++)
      {
        if(ts_id % Nthreads != ni)
          continue;
        auto& ts = testing_set[ts_id]; 
        networks[ni]->bind_input_layer(ts.first);
        networks[ni]->result = ts.second;
        networks[ni]->reset();
        rsums[ni] += networks[ni]->residual_sum();
      }
    }

    for(int ni=0; ni < Nthreads; ni++)
      rsum += rsums[ni];
    
    if(rsum > prsum)
      step *= -0.5;
    step_count ++;
  }
}
