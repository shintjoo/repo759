#include <iostream>
#include <cstdio>  // For printf
#include <cstdlib> // For atoi

int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc < 2) {
        std::cerr << "Error: Please provide an integer N as a command-line argument." << std::endl;
        return 1;
    }

    // Get the integer that is given in the command line (Assuming it is an integer)
    int N = std::atoi(argv[1]);

    //Print out each integer in ascending order with printf
    for (int i = 0; i <= N; i++){
        if (i > 0) {
            std::printf(" ");
        }
        std::printf("%d", i);
    }

    // Print out each integer in descending order using std::cout
    for (int i = N; i >= 0; --i) {
        if (i < N) {
            std::cout << " ";
        }
        std::cout << i;
    }
    std::cout << std::endl;

    return 0;
}