#include <iostream>
#include <fstream>
#include <sstream>
#include "sigmoid.h"
#include "network.h"
#include "MNISTreader.h"
#include <set>
#include <typeinfo>
#include <cmath>
#include <ctime>
#include <map>
#include <sys/stat.h>

void print_layer_topology(int l, std::vector<sigmoid_base*> layer)
{
  std::set<sigmoid_base*> next_layer;
  for(auto& o : layer)
  {
    for(int i=0; i!=l;i++)
      std::cout << ' ';
    std::cout << o->outputs.size() << " outs, " << o->inputs.size() << " inputs\n";
    if(typeid(*o) == typeid(sigmoid_smart))
    {
      for(auto& i : o->inputs)
        next_layer.insert(i.input_sigmoid);
    }
  }
  
  if(next_layer.size() == 0)
    return;
  std::vector<sigmoid_base*> nl;
  for(auto& it : next_layer)
    nl.push_back(it);
  print_layer_topology(l + 1, nl);
}

std::pair<int, int> get_stats(std::vector<network*>& nws, std::vector<std::pair<double*, double*>>& tss, double& rsum)
{
  int Nthreads = nws.size();
  std::vector<int> miss(Nthreads, 0);
  std::vector<int> hits(Nthreads, 0);
  std::vector<double> rsums(Nthreads, .0);

  #pragma omp parallel for
  for(int ni=0; ni < Nthreads; ni++)
  {
    for(int ts_id = 0; ts_id != tss.size(); ts_id++)
    {
      if(ts_id % Nthreads != ni)
        continue;
      auto& ts = tss[ts_id];
      nws[ni]->bind_input_layer(ts.first);
      nws[ni]->result = ts.second;
      nws[ni]->reset();
      nws[ni]->forward_run();
      rsums[ni] += nws[ni]->residual_sum();
      
      double max_out(.0);
      int result_id;
      for(int s_ptr_id(0); s_ptr_id != nws[ni]->output_layer_ptrs.size(); s_ptr_id++)
      {
        double v = nws[ni]->output_layer_ptrs[s_ptr_id]->value;
        if(v > max_out)
        {
          result_id = s_ptr_id;
          max_out = v;
        }
      }
      if(ts.second[result_id] == 1.0)
        hits[ni]++;
      else
        miss[ni]++;
    }
  }
  
  int _miss(0);
  int _hits(0);
  rsum = .0;
  for(int ni=0; ni < Nthreads; ni++)
  {
    _miss += miss[ni];
    _hits += hits[ni];
    rsum += rsums[ni];
  }

  return std::make_pair(_hits, _miss);
}

bool compare(double a, double b)
{
  if(std::abs(a) < 1e-6 && std::abs(b) < 1e-6)
    return true;
  if(std::abs(std::min(std::abs(a), std::abs(b)) / std::max(std::abs(a), std::abs(b)) - 1.0) > 1e-3)
    return false;
  if(a * b < .0)
    return false;
  return true;
}


void dump_network_to_file(const network& my_neural_network, std::string filename="nw.dat")
{
  std::ofstream nw_dump(filename);
  std::map<sigmoid_base*, int> ptr_to_id_map;
  for(int s_ptr_id=0; s_ptr_id != my_neural_network.all.size(); s_ptr_id++)
    ptr_to_id_map[my_neural_network.all[s_ptr_id]] = s_ptr_id;

  for(auto s_ptr : my_neural_network.all)
  {
    if(typeid(*s_ptr) == typeid(sigmoid_stupid))
    {
			nw_dump << "n " << ptr_to_id_map[s_ptr] << ' ' << s_ptr->inputs.size() << '\n';
			nw_dump << ptr_to_id_map[s_ptr->inputs[0].input_sigmoid] << ' ' << s_ptr->inputs[0].weight << '\n';
      //continue;
    }
    nw_dump << "n " << ptr_to_id_map[s_ptr] << ' ' << s_ptr->inputs.size() << '\n';
    for(auto i : s_ptr->inputs)
    {
      nw_dump << ptr_to_id_map[i.input_sigmoid] << ' ' << i.weight << '\n';
    }
  }
  nw_dump.close();
}

int main(int argc, char* argv[])
{
  int minibatch = 10;
  int Nthreads = 50;
  int chdirs_per_minibatch = 3;
  int learning_courses = 10;
  double learning_rate = 3.0;
  int neurons=30;
  std::string workdir="output";
  #define PARSE(ARG) if(name == #ARG) { sval >> ARG; continue;}
	#define PARSE2(ARG, VAR) if(name == #ARG) { sval >> VAR; continue;}
	for(int i=1;i<argc;i++)
	{
		std::string inp = std::string(argv[i]);
		size_t pos = inp.find("=");
		if(pos == std::string::npos)
			printf("you specified parameter wrong way use <name>=<value> format. NOTE: no \'-\' and spaces\n");
		else
		{
			std::string name = inp.substr(0,pos);
			std::stringstream sval;
			sval << inp.substr(pos+1,std::string::npos);
			printf("parameter[%d] has name %s and value %s\n",i-1,name.c_str(), sval.str().c_str());
			PARSE2(ths, Nthreads);
			PARSE2(mb, minibatch);
			PARSE2(lc, learning_courses);
			PARSE2(lr, learning_rate);
			PARSE2(cdpmb, chdirs_per_minibatch);
			PARSE2(wd, workdir);
			PARSE2(n, neurons);
			PARSE(Nthreads);
			PARSE(minibatch);
			PARSE(learning_courses);
			PARSE(learning_rate);
			PARSE(chdirs_per_minibatch);
			PARSE(workdir);
			PARSE(neurons);
		}
	}

  MNISTreader r("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
  //MNISTreader r("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
  // dump first 10 images of digits
  //dump first MNIST images to files:
  /*
  for(int i=0; i!=128; i++)
  {
    r.dump(i);
    std::cout << (unsigned int) r.getlabel(i) << '\n';
  }
  */
  
	if(r.size() % minibatch != 0)
	{
		while(r.size() % minibatch != 0)
			minibatch++;
		std::cout << "chosing minibatch as closest devisor of trainig data size. new minibatch is " << minibatch << '\n';
	}

  if(Nthreads > minibatch)
  {
		std::cout << "Reducing Nthreads to " << minibatch << '\n';
		Nthreads = minibatch;
	}

	if(workdir == "output")
	{
		workdir="out_mb"+std::to_string(minibatch)+
						"_lc"+std::to_string(learning_courses)+
						"_cd"+std::to_string(chdirs_per_minibatch)+
						"_n"+std::to_string(neurons)+
						"_lr"+std::to_string((int) floor(learning_rate));
	}
	
	struct stat st;
	if(stat(workdir.c_str(), &st) == 0)
	{
    if(st.st_mode & S_IFDIR != 0)
    {
			std::cerr << "working directory exists. exiting\n";
			return -1;
		}
	}
	else
	{
		if(mkdir(("./"+workdir).c_str(), 0755) != 0)
		{
			std::cerr << "cannot create directory for ouput. exiting\n";
			return -1;
		}
	}
	
	std::ofstream info((workdir+'/'+"info.log").c_str());
	info << "minibatch=" << minibatch << '\n';
	info << "chdirs_per_minibatch=" << chdirs_per_minibatch << '\n';
	info << "learning_rate=" << learning_rate << '\n';
	info << "learning_courses=" << learning_courses << '\n';
	info << "neurons=" << neurons << '\n';
	info.close();
  
  // create network
  int rand_initializer=13;
  
  std::vector<network*> networks;
  for(int i=0; i!=Nthreads; i++)
  {
    srand(rand_initializer);
    network* nn = new network;
    nn->make_input_layer(r.getimage(0).size());
    nn->add_layer(neurons);
    nn->add_layer(10);
    nn->bake();
    networks.push_back(nn);
  }
  
  //#define TESTS
  //note tests will fail for input neurons and it's weights due to zero partial derivative hardcoded in network.cpp network::calculate_all_derivatives()
  network& my_neural_network = *(networks[0]);
  
  /*
   * below is the unit test for network::forward_run() and network::get_derivatives_as_vector()
   */
  #ifdef TESTS
	std::cout << "total number of neurons is " << my_neural_network.all.size() << '\n';
	int answer = r.getlabel(0);

	std::cout << "answer is " << answer << '\n';
	
	std::vector<double> testing_input;
	testing_input.resize(my_neural_network.input_layer_ptrs.size());
	
	networks[0]->bind_input_layer(testing_input.data());
	
	for(int i=0; i!=3; i++)
	{
		for(auto& it : testing_input)
			it = 0.3 + 0.7 * (double) rand() / (double) RAND_MAX;

		sigmoid_globals::reset_fire_count();
		my_neural_network.forward_run();
		std::cout << "================fire_cout:" << sigmoid_globals::get_fire_count() << '\n';
		
		sigmoid_globals::reset_fire_count();
		double diff(.0);
		for(auto& out : my_neural_network.output_layer_ptrs)
			diff += pow(out->fire(), 2.0) - pow(out->value, 2.0);
		
		std::cout << "================fire_cout:" << sigmoid_globals::get_fire_count() << '\n';

		std::cout << "diff=" << diff << '\n';
	}
	
	std::cout << "testing derivative\n";
	
	//=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======
	//=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======
	//=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======DERIVATIVE=======
	
	std::vector<double> test_result;
	test_result.resize(my_neural_network.output_layer_ptrs.size());
	for(auto& r : test_result)
		r = .0;
	//test_result[answer] = 1.0;
	my_neural_network.result = test_result.data();
	my_neural_network.calculate_all_derivatives();
	my_neural_network.get_derivatives_as_vector();

	std::map<std::pair<int,int>, int> toderivatives;
	std::map<int, std::pair<int,int>> fromderivatives;
	size_t count(0);
	for(size_t s_ptr_id(0); s_ptr_id != my_neural_network.all.size(); s_ptr_id++)
	{
		auto s_ptr = my_neural_network.all[s_ptr_id];
		size_t i(0);
		for(; i != s_ptr->inputs.size(); i++)
		{
			toderivatives[std::make_pair(s_ptr_id, i)] = count;
			fromderivatives[count] = std::make_pair(s_ptr_id, i);
			count ++;
		}
		toderivatives[std::make_pair(s_ptr_id, i)] = count;
		fromderivatives[count] = std::make_pair(s_ptr_id, i);
		count ++;
	}
	
	count=0;
	for(size_t s_ptr_id(0); s_ptr_id != my_neural_network.all.size(); s_ptr_id++)
	{
		auto s_ptr = my_neural_network.all[s_ptr_id];
		size_t i(0);
		for(; i != s_ptr->inputs.size(); i++)
		{
			if(*(my_neural_network.derivatives[count].first) != s_ptr->inputs[i].weight)
				std::cout << "WROOONG - v! - w\n";
			if(my_neural_network.derivatives[count].first != &(s_ptr->inputs[i].weight))
				std::cout << "WROOONG - p! - w\n";
			count ++;
		}
		if(*(my_neural_network.derivatives[count].first) != s_ptr->bias)
			std::cout << "WROOONG - v! - b\n";
		if(my_neural_network.derivatives[count].first != &(s_ptr->bias))
			std::cout << "WROOONG - p! - b\n";
		count ++;
	}
	
	//===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test
	//===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test
	//===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test===DERIVATIVE-test
	
	//srand(time(NULL));
	bool failed(false);
	#define DERIV_TESTS_COUNT 33
	for(int tn(0); tn != DERIV_TESTS_COUNT; tn++)
	{
		double deriv_direct_prev = -1e10;
		double deriv_direct = deriv_direct_prev;
		double dweight(1.0);
		int step_count(0);
		int deriv_id = rand() % my_neural_network.derivatives.size();  
		double deriv = my_neural_network.derivatives[deriv_id].second;
		while(true || step_count != 1000)
		{
			step_count++;
			my_neural_network.reset();
			double s0 = my_neural_network.residual_sum();
			double original = *(my_neural_network.derivatives[deriv_id].first);
			if(dweight < 1e-9 * std::abs(original))
				break;
			*(my_neural_network.derivatives[deriv_id].first) += dweight;
			my_neural_network.reset();
			double s1 = my_neural_network.residual_sum();
			*(my_neural_network.derivatives[deriv_id].first) = original;
			deriv_direct = (s1 - s0) / dweight;
			dweight *= 0.5;
			if(std::abs(deriv_direct_prev - deriv_direct) <= 1e-8 * std::max(std::abs(deriv_direct), std::abs(deriv_direct_prev)) && step_count > 10)
				break;
			deriv_direct_prev = deriv_direct;
			if(step_count == 1000)
				std::cerr << "ERROR: no convergence!\n";
		}
		
		
		
		
		if(fromderivatives.find(deriv_id) == fromderivatives.end())
			std::cerr << "cannot find derivative " << deriv_id << " in fromderivatives\n";
		
		int deriv_id1 = fromderivatives[deriv_id].first;
		
		int deriv_id2 = fromderivatives[deriv_id].second;


		//std::cout << "[" << deriv_id << "]direct result is " << deriv_direct << ". deriv[" << deriv_id1 << ", " << deriv_id2 << "] is " << deriv << '\n';

		if(!compare(deriv, deriv_direct))
		{
			double deriv_direct_prev = -1e10;
			double deriv_direct2 = deriv_direct_prev;
			double dweight(1.0);
			int step_count2(0);
			
			
			while(true || step_count2 != 1000)
			{
				step_count2++;
				my_neural_network.reset();
				double s0 = my_neural_network.residual_sum();
				if(deriv_id2 != my_neural_network.all[deriv_id1]->inputs.size())
				{
					double weight_original = my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight;
					if(dweight < 1e-9 * std::abs(weight_original))
						break;
					my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight += dweight;
					my_neural_network.reset();
					double s1 = my_neural_network.residual_sum();
					my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight = weight_original;
					deriv_direct2 = (s1 - s0) / dweight;
				}
				else
				{
					double bias_original = my_neural_network.all[deriv_id1]->bias;
					if(dweight < 1e-9 * std::abs(bias_original))
						break;
					my_neural_network.all[deriv_id1]->bias += dweight;
					my_neural_network.reset();
					double s1 = my_neural_network.residual_sum();
					my_neural_network.all[deriv_id1]->bias = bias_original;
					deriv_direct2 = (s1 - s0) / dweight;
				}
				dweight *= 0.5;
				if(std::abs(deriv_direct_prev - deriv_direct2) <= 1e-8 * std::max(std::abs(deriv_direct2), std::abs(deriv_direct_prev)) && step_count2 > 10)
					break;
				deriv_direct_prev = deriv_direct2;
				if(step_count2 == 1000)
					std::cerr << "ERROR: no convergence!\n";
				//std::cout << "direct result is " << deriv_direct << '\n';
			}

			
			std::cout << "#" << tn << " id(" << deriv_id << ") failed: " << deriv << " != " << deriv_direct << " (step count = " << step_count << ")[" << deriv_id << "], deriv_direct2 is " << deriv_direct2 << "(step count = " << step_count2 << ")[" << deriv_id1 << ", " << deriv_id2 << "]\n";
			failed = true;
		}
	}
	std::cout << "-----------------\n";
	
	//===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii
	//===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii
	//===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii===DERIVATIVE-test_ii
	
	for(int tn(0); tn != DERIV_TESTS_COUNT; tn++)
	{
		int deriv_id1 = rand() % my_neural_network.all.size();  
			
		int deriv_id2 = rand() % (my_neural_network.all[deriv_id1]->inputs.size() + 1);  
		double deriv(.0);

		deriv = my_neural_network.all[deriv_id1]->derivatives[deriv_id2];
		//direct derivative calculation

		if(deriv != my_neural_network.derivatives[toderivatives[std::make_pair(deriv_id1, deriv_id2)]].second)
			std::cout << "ACHTUNG!!!\n";
		
		double deriv_direct_prev = -1e10;
		double deriv_direct = deriv_direct_prev;
		double dweight(1.0);
		int step_count(0);
		while(true || step_count != 1000)
		{
			step_count++;
			my_neural_network.reset();
			double s0 = my_neural_network.residual_sum();
			if(deriv_id2 != my_neural_network.all[deriv_id1]->inputs.size())
			{
				double weight_original = my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight;
				if(dweight < 1e-9 * std::abs(weight_original))
					break;
				my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight += dweight;
				my_neural_network.reset();
				double s1 = my_neural_network.residual_sum();
				my_neural_network.all[deriv_id1]->inputs[deriv_id2].weight = weight_original;
				deriv_direct = (s1 - s0) / dweight;
			}
			else
			{
				double bias_original = my_neural_network.all[deriv_id1]->bias;
				if(dweight < 1e-9 * std::abs(bias_original))
					break;
				my_neural_network.all[deriv_id1]->bias += dweight;
				my_neural_network.reset();
				double s1 = my_neural_network.residual_sum();
				my_neural_network.all[deriv_id1]->bias = bias_original;
				deriv_direct = (s1 - s0) / dweight;
			}
			dweight *= 0.5;
			if(std::abs(deriv_direct_prev - deriv_direct) <= 1e-8 * std::max(std::abs(deriv_direct), std::abs(deriv_direct_prev)) && step_count > 10)
				break;
			deriv_direct_prev = deriv_direct;
			if(step_count == 1000)
				std::cerr << "ERROR: no convergence!\n";
			//std::cout << "direct result is " << deriv_direct << '\n';
		}
		//std::cout << "direct result is " << deriv_direct << ". deriv[" << deriv_id1 << ", " << deriv_id2 << "] is " << deriv << '\n';
		if(!compare(deriv, deriv_direct))
		{
			std::cout << "#" << tn << " failed: " << deriv << " != " << deriv_direct << " (step count = " << step_count << ", neuron #" << deriv_id1 << "(" << my_neural_network.all[deriv_id1]->inputs.size() << "), parameter #" << deriv_id2 << ")" << "\n";
			failed = true;
		}
		//else
		//  std::cout << "#" << tn << " passed\n";
	}
	
	//===derivative_END===derivative_END===derivative_END===derivative_END
	//===derivative_END===derivative_END===derivative_END===derivative_END
	//===derivative_END===derivative_END===derivative_END===derivative_END
	
	if(!failed)
		std::cout << "\tderivation tests are done\n";
	else
		std::cout << "\tderivation tests FAILED!\n";
  #endif
  
  std::vector<std::vector<double>> answers;
  answers.resize(r.size());
  for(int i=0; i!= r.size(); i++)
  {
    const int ans_sz = my_neural_network.output_layer_ptrs.size();
    answers[i].resize(ans_sz);
    int ans = r.getlabel(i);
    for(int j=0; j!= ans_sz; j++)
      answers[i][j] = j == ans ? 1.0 : .0;
  }

  std::ofstream dump((workdir+'/'+"out.dat").c_str());

  for(int learning_course(0); learning_course != learning_courses; learning_course++)
  {
		//srand(13);
    int overall_hits(0);
    int overall_miss(0);
    std::set<int> all_learning_items;
    for(int i=0; i!=r.size(); i++)
      all_learning_items.insert(i);
    int sub_size = minibatch;
    for(int mb_id=0; mb_id * sub_size < r.size(); mb_id++)
    {
      std::set<int> ts;
      for(int j=0; j!=sub_size; j++)
      {
        while(true)
        {
          int picked = rand() % all_learning_items.size();
          std::set<int>::iterator it = all_learning_items.begin();
          for(int _i=0; _i!=picked; _i++, it++);
          //std::cout << "chosen label " << r.getlabel(*it) << '\n';
          bool same = false;
          for(auto k: ts)
          {
            if(r.getlabel(k) == r.getlabel(*it))
              same = true;
          }
          //if(same)
          //  continue;
        
          ts.insert(*it);
          all_learning_items.erase(*it);
          break;
        }
        
        //ts.insert(mb_id * sub_size + j);
      }
      
      std::vector<int> histogram;
      histogram.resize(my_neural_network.output_layer_ptrs.size());
      for(auto& it : histogram)
        it = 0;
      for(auto item_id : ts)
      {
        for(int j=0; j!=answers[item_id].size(); j++)
          histogram[j] += answers[item_id][j];
      } 
      std::cout << "learning course # " << learning_course << ", " << (floor) ((double) mb_id * sub_size / (double) r.size() * 100.0) << "% , histogram is:";
      for(auto& it : histogram)
        std::cout << " " << it;
      std::cout << "   success: ";
      
      std::vector<std::pair<double*, double*>> testing_data_pointers;
      for(auto item_id : ts)
        testing_data_pointers.push_back(std::make_pair(r.getimage(item_id).data(), answers[item_id].data()));
      
      //std::cout << "descending\n";
      double rsum(.0);

      std::pair<int, int> hits_miss;

      hits_miss = get_stats(networks, testing_data_pointers, rsum);
      std::cout /*<< "rsum=" << rsum */<<", hits=" << hits_miss.first << ", miss=" << hits_miss.second << " -> ";
      dump << learning_course << '\t' << hits_miss.first << '\t' << hits_miss.second << '\n';
      overall_hits += hits_miss.first;
      overall_miss += hits_miss.second;

      //dump mini-batch to file
      /*
      {
        int ts_id(0);
        for(auto item_id : ts)
        {
          auto& ts = testing_data_pointers[ts_id];
          std::ofstream of("char_"+std::to_string(ts_id)+".dat");
          for(int i(0); i != my_neural_network.input_layer_ptrs.size(); i++)
            of << ts.first[i] << '\n';
          for(int i(0); i != my_neural_network.output_layer_ptrs.size(); i++)
            of << ts.second[i] << '\n';
          of << r.getlabel(item_id) << '\n'; 
          of.close();
          ts_id++;
        }
      }
      */

      for(int i=0;i!=chdirs_per_minibatch;i++)
      {
        my_neural_network.descent_single_direction(testing_data_pointers, networks, learning_rate);
        hits_miss = get_stats(networks, testing_data_pointers, rsum);
      }
      std::cout /*<< "rsum=" << rsum */<< ", hits=" << hits_miss.first << ", miss=" << hits_miss.second << '\n';
      //std::cout << "\tdescending is done!\n";
      //break;
    }
    dump_network_to_file(my_neural_network, (workdir+'/'+"nw"+std::to_string(learning_course)+".dat").c_str());
    std::cerr << "THIS " << learning_course << " LEARNING COURSE SUCCESS:\n\ttotal hits : " << overall_hits << "(" << floor((double) overall_hits / ((double) overall_hits + (double) overall_miss) * 100.0) << "%)\n\ttotal miss : " << overall_miss << "(" << floor((double) overall_miss / ((double) overall_hits + (double) overall_miss) * 100.0) << "%)\n\n";
  }
  dump.close();
  return 0;
}
