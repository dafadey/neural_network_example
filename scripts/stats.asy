//string path=".";
import graph;

pen LAB(real l, real a, real b)
{
  real Y1 = (l+16)/116;
  real X1 = a/500 + Y1;
  real Z1 = -b/200 + Y1;
  
  X1 = X1 > 0.206893 ? X1^3 : (X1-16/116)/7.787;
  Y1 = Y1 > 0.206893 ? Y1^3 : (Y1-16/116)/7.787;
  Z1 = Z1 > 0.206893 ? Z1^3 : (Z1-16/116)/7.787;
  
  real X = X1;
  real Y = Y1;
  real Z = Z1;
  
  real r = 2.0413690 * X -0.5649464 * Y -0.3446944 * Z;
  real g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  real b = 0.0134474 *X + -0.1183897 * Y + 1.0154096 * Z;
  
  return r * red + g * green + b * blue; 
}

file in=input(path+"/out.dat").line();
string[] lines = in;

real[][] lc;
int[] total_hits;
int[] total_miss;
for(string l : lines)
{
  string[] items = split(l, '\t');
  while(lc.length < (int) items[0] + 1)
  {
    total_hits.push(0);
    total_miss.push(0);
    lc.push(new real[]);
  }
  real v = 100 * (real) items[1] / ((real) items[1] + (real) items[2]);
  int id = (int) items[0];
  lc[id].push(v);
  total_hits[id] += (int) items[1];
  total_miss[id] += (int) items[2];
  //write("adding " + string(v) + ": " + items[1] + ":" + items[2] + " to lc#"+string(id));
}
srand(11);
picture p;
pen[] colors;
for(real[] data : lc)
{
  guide g;
  for(int v_i=0; v_i!=data.length; ++v_i)
    g  =g--(v_i, data[v_i]);
  //real alpha = unitrand() * pi * 2.0;
  //pen cl = LAB(60, cos(alpha) * 64, sin(alpha) * 64);
  pen cl = hsv(unitrand() * 360, 0.9, 0.7);
  colors.push(cl);
  
  draw(p, g, cl);
}

ylimits(p,0,100);
xaxis(p, "bunch id", BottomTop, LeftTicks);
yaxis(p, "\%", LeftRight, RightTicks);
size(p, 8cm, 6cm, point(p, SW), point(p, NE));

add(p.fit());

picture ptot;
guide gt;
pair v;
for(int i=0; i!=total_hits.length; ++i)
{
	v=(i, 100*total_hits[i]/(total_hits[i]+total_miss[i]));
  gt=gt--v;
}
draw(ptot, gt);
label(ptot,string(v.y)+"\,\%" ,v, E);
dot(ptot, gt, 0.5*green);
ylimits(ptot,0,100);
xaxis(ptot, "bunch id", BottomTop, LeftTicks);
yaxis(ptot, "\%", LeftRight, RightTicks);
size(ptot, 8cm, 6cm, point(ptot, SW), point(ptot, NE));

add(ptot.fit(), (10cm,0));
