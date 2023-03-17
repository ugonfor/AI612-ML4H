#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

using namespace std;


int main(int argc, char const *argv[])
{
    ifstream infile;
    infile.open("./dataset/CHARTEVENTS.csv");
    ofstream outfile;
    outfile.open("./filtered_dataset/CHARTEVENTS.csv");

    string tmp;
    for (int i = 0; i < 100000; i++)
    {
        getline(infile, tmp);
        outfile << tmp << "\n";
    }
    
    return 0;
}
