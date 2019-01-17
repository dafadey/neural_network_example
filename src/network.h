#pragma once
#include "sigmoid.h"

struct network
{
  network() : all(), input_layer_ptrs(), output_layer_ptrs(), result(nullptr), fired(false), derivatives() {}
  
  void make_input_layer(size_t /*n*/); // creates n sigmod_stupid instances
  void bind_input_layer(double* /*arr*/); // binds a chunk of double-s to input layer
  void add_layer(int); // number of layers
  void make_forward_connections(); // fills
  void bake(); // actually the wrapper that builds all connections and collect all sigmoids into one container
  
  
  void reset();
  
  void forward_run(); // fast calculation of values on the output sigmoids of network
  
  double residual_sum();
  
  /* 
   * Three methods to calcuate derivative and put it into linear vector 'derivatives'
   * Expected to be called one by one
   */
  void prepare_derivative();
  void calculate_all_derivatives();
  void get_derivatives_as_vector();
  
  /*
   * This method performs learning step. Multiple adaptive steps are available by changin MAX_STEP_COUNT macro. see implementation in network.cpp
   */
  void descent_single_direction(std::vector<std::pair<double* /*input images*/, double* /*ouput answers array*/>>&, std::vector<network*>&, double /*learning_rate*/ = 3.0);
  
  //data
  std::vector<sigmoid_base*> all; // we need this for broadcast operations like invalidation of whole network (make sigmoid values invalid)
  
  std::vector<sigmoid_stupid*> input_layer_ptrs;
  std::vector<sigmoid_smart*> output_layer_ptrs;
  
  double* result; // size of output_layer. bit mask of right answer

  bool fired;
 
  std::vector<std::pair<double*, double>> derivatives;

};
