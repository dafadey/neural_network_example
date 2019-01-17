namespace sigmoid_globals
{
  static long fire_count = 0;
  
  long get_fire_count()
  {
    return fire_count;
  }
  
  void inc_fire_count()
  {
    fire_count++;
  }
  
  void reset_fire_count()
  {
    fire_count = 0;
  }
}
