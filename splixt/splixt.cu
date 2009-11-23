#include <splixt_kernel.cu>
#include <splixt_impl.cpp>

int main(int argc, char** argv)
{
	runSplitText(argc, argv);
    //runCCA(argc, argv);
    cutilExit(argc, argv);
}