#pragma once
#include <vector>
#include <cstdlib>
#include "sigmoid_globals.h"
//#include <limits>

//#define ACCURATESUM
/*
use this flag to group all negative and positive values sum them in modulus accesnfing order and finally sum accumulated negative and positive parts.
it works very slow. feature added to test if round-off errors influence badly on learning process.
*/

struct summator
{
	std::vector<double> positives;
	std::vector<double> negatives;
	void MADD(double a);
	double MR();
};

struct sigmoid_base;

struct sigmoid_input
{
  sigmoid_base* input_sigmoid;
  double weight;
};

struct sigmoid_base
{
  #define INVALID_SIGMOID -1.0
  sigmoid_base() : inputs(), outputs(),
                   bias(.0), value(INVALID_SIGMOID),
                   derivative_factor(.0),
                   back_fires_count(0) {} 
//methods
  double self(double) const;
  double self_arg(double) const;
  void add_outputs();
  virtual double fire() const = 0;
  virtual bool calculate_value() = 0; // return true if sigmoid was got it's value. sigmoid can get its value if all inputs are linked ot VALID sigmoids
  virtual bool calculate_derivative_value() = 0;
  void invalidate_derivative()
  {
    derivative_factor = .0;
    back_fires_count = 0;
  }
  void invalidate()
  {
    value = INVALID_SIGMOID;
    invalidate_derivative();
  }
//data
  std::vector<sigmoid_input> inputs;
  std::vector<sigmoid_base*> outputs;
  std::vector<double> derivatives;
  double bias;
  double value;
  double derivative_factor;
  int back_fires_count; // used for derivative calculation
};

struct sigmoid_smart : public sigmoid_base
{
  virtual double fire() const;
  virtual bool calculate_value();
  virtual bool calculate_derivative_value();
};

struct sigmoid_stupid : public sigmoid_base
{
  virtual double fire() const;
  virtual bool calculate_value();
  virtual bool calculate_derivative_value();
};
