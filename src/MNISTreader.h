#pragma once

#include <vector>

struct MNISTitem
{
  int label;
  std::vector<double> image;
};

struct MNISTreader
{
  MNISTreader(const char* MNIST_image_file, const char* MNIST_label_file);
  
  std::vector<double>& getimage(int item_id)
  {
    return data[item_id].image;
  }
  
  int getlabel(int item_id) const
  {
    return data[item_id].label;
  }
  
  int size() {return data.size();}
  
  void dump(int);
  
  std::vector<MNISTitem> data;
};
