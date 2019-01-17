//string infile="nw.dat";
import graph;
import palette;

file in=input(infile).line();
string[] lines = in;

struct weight
{
  int id;
  real w;
};

struct neuron
{
  int id;
  weight[] weights;
};

neuron[] neurons;

bool order_error = false;
for(int l_id=0; l_id < lines.length; ++l_id)
{
  string[] items = split(lines[l_id], ' ');
  if(items[0] == 'n')
  {
    neuron nn;
    nn.id = (int) items[1];
    int n = (int) items[2];
    nn.weights = new weight[n];
    //write("reading neuron " + string(nn.id) + " with " + string(n) + " weights");
    for(int i=0; i!=n; ++i)
    {
      l_id+=1;
      string[] _items = split(lines[l_id], ' ');
      weight w;
      w.w = (real) _items[1];
      w.id = (int) _items[0];
      nn.weights[i] = w;
      if(i==0)
        continue;
      if(split(lines[l_id-1], ' ')[0] >= _items[0])
        order_error = true;
    }
    neurons.push(nn);
  }
}
write("read " + string(neurons.length) + " neurons");

pen mycolor(real x)
{
  return x > 0 ? abs(x) * red : abs(-x) * blue + x^2 * green;
}

pen[] mypal(int n)
{
  pen[] res = new pen[n];
  for(int i=0; i!=n; ++i)
  {
    real x = 2.0 * (i / (n - 1) - 0.5);
    res[i] = mycolor(x);
  }
  return res;
}

int[] outs;

for(neuron nn : neurons)
{
  if(nn.weights.length < 700 & nn.weights.length > 1)
    outs.push(nn.id);
}

int choice=2;

int find_by_id(int id)
{
  for(int nn_i=0; nn_i != neurons.length; ++nn_i)
  {
    if(neurons[nn_i].id == id)
      return nn_i;
  }
  return -1;
}

int target_neuron = find_by_id(outs[choice]);

real rc=0.5cm;
real Li=3cm;
real space=0.3cm;
real wmax = 0.0; 
for(weight w : neurons[target_neuron].weights)
  wmax = max(wmax, abs(w.w));


real[][] avg_pattern = new real[28][28];
for(int i=0 ; i!= 28*28; ++i)
  avg_pattern[i % 28][floor(i / 28)] = .0;

int N = neurons[target_neuron].weights.length;
int NX = floor(sqrt(N));

int ix=0;
int jx=0;
for(weight w : neurons[target_neuron].weights)
{
  real alpha = (w.id % N) / N * 2.0 * pi;
  filldraw(shift((Li+space) * ix, jx * (Li +space+ 2*rc)-rc) * scale(rc) * unitcircle, mycolor(w.w / wmax), black);
  int i = find_by_id(w.id);
  int dim = floor(sqrt(neurons[i].weights.length));
  real[][] pattern = new real[dim][dim];
  for(int w_id=0; w_id!=neurons[i].weights.length; ++w_id)
  {
    pattern[floor(w_id / dim)][w_id % dim] = neurons[i].weights[w_id].w;
    //avg_pattern[floor(w_id / dim)][w_id % dim] += neurons[i].weights[w_id].w * w.w;
  }
  picture p;
  image(p, pattern, (0,0), (dim,dim), mypal(1024));
  size(p, Li, Li, point(p,SW), point(p,NE));
  add(p.fit(), ((Li+space)*ix-(Li+space)/2, (Li+space+2*rc)*jx));
  ix+=1;
  if(ix==NX)
  {
    ix=0;
    jx+=1;
  }
}

/*
picture p;
image(p, avg_pattern, (0,0), (28,28), mypal(1024));
picture p1 = shift(-28/2, -28/2) * p;
size(p1, Li*3, Li*3, point(p1,SW), point(p1,NE));
add(p1.fit(), (0,0));
*/
