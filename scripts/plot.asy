import palette;

void draw_letter(int i, pair pos)
{
  //write("reading "+"char_"+string(i)+".dat");
  file inp = input("char_"+string(i)+".dat");
  real[] data = inp;
  write(data.length);
  int n = 28;//floor(sqrt(data.length));
  real[][] img = new real[n][n];
  write(n);
  for(int i=0; i!=n; ++i)
  {
    for(int j=0; j!=n; ++j)
      img[j][n-1-i] = 1.0 - data[i*n+j];
  }
  real[] res=new real[10];
  for(int i=0; i!=10; ++i)
    res[i] = data[n*n+i];
  picture p;
  image(p, img, (0,0), (28,28), Grayscale());
  size(p, 1cm, 1cm, point(p, SW), point(p, NE));
  add(p.fit(), pos);
  for(int count=0; count !=10; ++count)
  {
    pair cp = pos+(0.05cm + count*0.09cm)+(0,1.1cm);
    fill(shift(cp)*scale(0.03cm)*unitcircle, red * res[count]);
  }
  label(string(data[n*n+10]), pos+(0.5cm, 1.2cm), N);
}

int N=128;
int n = floor(sqrt(N));
for(int j=0; j!=floor(N/n); ++j)
{
	for(int i=0; i!=n; ++i)
	{
		write("drawing ",i+j*n);
		if(i+j*n<N)
			draw_letter(i+j*n, (i*1.3cm,j*2.1cm));
	}
}
