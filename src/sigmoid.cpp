#include "sigmoid.h"
#include <iostream>
#include <set>
#include <typeinfo>
#include <cmath>
#include <algorithm>

void summator::MADD(double v)
{
	if(v > .0)
		positives.push_back(v);
	else
		negatives.push_back(v);
}

double summator::MR()
{
		std::sort(positives.begin(), positives.end(), [](const double& a, const double& b) -> bool {return std::abs(a) < std::abs(b);});
		std::sort(negatives.begin(), negatives.end(), [](const double& a, const double& b) -> bool {return std::abs(a) < std::abs(b);});
		double P(.0);
		for(auto v : positives)
			P += v;
		double N(.0);
		for(auto v : negatives)
			N += v;
		positives.clear();
		negatives.clear();
		return P + N;
}

/*
 * Main sigmoid function
 */
double sigmoid_base::self(double s) const
{
  sigmoid_globals::inc_fire_count();
  return 1.0 / (1.0 + exp(-s));
}

/*
 * Derivative of self with respect to its argument
 */
double sigmoid_base::self_arg(double s) const
{
  return exp(-s) / ((1.0 + exp(-s)) * (1.0 + exp(-s)));
}

void sigmoid_base::add_outputs()
{
  for(auto& i : inputs)
  {
    bool skip(false);
    for(auto& o : i.input_sigmoid->outputs)
    {
      if(o == this)
        skip = true;
    }
    if(skip)
      continue;
    i.input_sigmoid->outputs.push_back(this);
    if(typeid(*(i.input_sigmoid)) == typeid(sigmoid_smart))
      i.input_sigmoid->add_outputs();
  }
}

double sigmoid_smart::fire() const
{
  double s(.0);
  for(const auto& i : inputs)
    s += i.input_sigmoid->fire() * i.weight;
  return self(s + bias);
}

bool sigmoid_smart::calculate_value()
{
  double s(.0);
	#ifdef ACCURATESUM
	summator sum;
	#endif
  for(const auto& i : inputs)
  {
    if(i.input_sigmoid->value == INVALID_SIGMOID)
      return false;
    double v = i.input_sigmoid->value * i.weight;
		#ifdef ACCURATESUM
		sum.MADD(v);
		#else
    s += v;
    #endif
  }
	#ifdef ACCURATESUM
	s += sum.MR();
	#endif
  value = self(s + bias);
  return true;
}

bool sigmoid_smart::calculate_derivative_value()
{
  double s(.0);
	#ifdef ACCURATESUM
	summator sum;
	#endif
  for(const auto& i : inputs)
  {
    if(i.input_sigmoid->value == INVALID_SIGMOID)
      return false;
    double v = i.input_sigmoid->value * i.weight;
		#ifdef ACCURATESUM
		sum.MADD(v);
		#else
    s += v;
    #endif
  }
  #ifdef ACCURATESUM
	s += sum.MR();
	#endif
  derivative_factor = self_arg(s + bias);
  return true;
}

double sigmoid_stupid::fire() const
{
  double s(.0);
  for(const auto& i : inputs)
    s += *((double*) i.input_sigmoid) * i.weight;
  return self(s + bias);
}

bool sigmoid_stupid::calculate_value()
{
  double s(.0);
  for(const auto& i : inputs)
    s += *((double*) i.input_sigmoid) * i.weight;
  value = self(s + bias);
  return true;
}

bool sigmoid_stupid::calculate_derivative_value()
{
  double s(.0);
  for(const auto& i : inputs)
    s += *((double*) i.input_sigmoid) * i.weight;
  derivative_factor = self_arg(s + bias);
  return true;
}
