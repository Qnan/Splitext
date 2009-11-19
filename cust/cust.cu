#include <cust_kernel.cu>
#include <cust_impl.cpp>

int main(int argc, char** argv)
{
   //runTest(argc, argv);
   runCCA(argc, argv);
   cutilExit(argc, argv);
}