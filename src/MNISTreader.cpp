#include "MNISTreader.h"
#include <fstream>
#include <iostream>

static uint32_t read32(std::ifstream& i)
{
  unsigned char bytes[4];
  i.read(reinterpret_cast<char*>(bytes), 4);
  uint32_t res;
  res = ((uint32_t) bytes[3]) | (((uint32_t) bytes[2]) << 8) | (((uint32_t) bytes[1]) << 16) | (((uint32_t) bytes[0]) << 24);
  //std::cout << "------ " << std::hex << (uint32_t) bytes[0] << ' ' << std::hex << (uint32_t) bytes[1] << ' ' << std::hex << (uint32_t) bytes[2] << ' ' << std::hex << (uint32_t) bytes[3] << " = " << res << '\n';
  return res;
}

static uint8_t read8(std::ifstream& i)
{
  unsigned char byte;
  i.read(reinterpret_cast<char*>(&byte), 1);
  return (uint32_t) byte;
}

MNISTreader::MNISTreader(const char* MNIST_image_file, const char* MNIST_label_file)
{
  uint32_t w32;
  uint16_t w16;
  uint8_t w8;
  std::ifstream imf(MNIST_image_file, std::ios::binary);
  std::ifstream lbf(MNIST_label_file, std::ios::binary);
  
  // check magic word
  bool sofarsogood = true;
  w32 = read32(imf);
  if(w32 != 0x00000803)
  {
    std::cerr << "MNIST reader ERROR: image file has wrong signature, expected " << 0x00000803 << ", got " << std::hex << w32 << '\n';
    sofarsogood = false;
  }
  w32 = read32(lbf);
  if(w32 != 0x00000801)
  {
    std::cerr << "MNIST reader ERROR: label file has wrong signature, expected " << 0x00000801 << ", got " << std::hex << w32 << '\n';
    sofarsogood = false;
  }
  
  // reading number of items
  long num(0);
  w32 = read32(imf);
  if(read32(lbf) == w32)
  {
    num = w32;
    std::cout << "MNIST reader INFO: items to read is " << num << '\n';
  }
  else
  {
    std::cerr << "MNIST reader ERROR: cannot get number of items\n";
    sofarsogood = false;
  }
  // read rows, cols
  long nrow(read32(imf));
  long ncol(read32(imf));
  std::cout << "MNIST reader INFO: number rows is " << nrow << ", number of columns is " << ncol << '\n';
  for(long j(0); j != num; j++)
  {
    MNISTitem it;
    it.label = (int) read8(lbf);
    it.image.resize(nrow * ncol);
    for(long i(0); i != nrow * ncol; i++)
      it.image[i] = (double) read8(imf) / 256.0;
    
    data.emplace_back(it);
  }
  std::cout << "MNIST reader INFO: read " << data.size() << " items \n";
  
  lbf.close();
  imf.close();
}

void MNISTreader::dump(int i)
{
  std::ofstream f(("char_"+std::to_string(i)+".dat").c_str());
  for(const auto& pixel : data[i].image)
    f << pixel << '\n';
  for(int c(0); c != 10; c++)
    f << (data[i].label == c ? 1.0 : 0.0) << '\n';
  f << data[i].label << '\n';
  f.close();
}
