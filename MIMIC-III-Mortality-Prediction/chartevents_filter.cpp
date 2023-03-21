// g++ -O2 -o chartevents_filter ./chartevents_filter.cpp
// ./chartevents_filter ./dataset/CHARTEVENTS.csv ./filtered_dataset/CHARTEVENTS.csv ./filtered_dataset/ICUSTAY_ID_TIME_PAIR.csv

#include <iostream>
#include <fstream>
#include <string>

#include <stdint.h>
#include <time.h>
#include <ctime>
#include <iomanip>

#include <signal.h>

#include <map>


using namespace std;

map<int, string> ICUSTAY_ID_TIME_Map;

ifstream infile;
ofstream outfile;

void sigint_handler(int sig){
    infile.close();
    outfile.close();
    exit(-1);
}

bool init_filter_pair(const char* path){

    ifstream filter_file;
    filter_file.open(path);

    string line;
    while (getline(filter_file, line))
    {
        int id = stoi(line.substr(0, line.find(",")));
        string timestamp = line.substr(line.find(",") + 1);
        
        //printf("id: %d / timestamp : %s\n", id, timestamp.c_str());

        ICUSTAY_ID_TIME_Map.insert(make_pair(id, timestamp));
    }
    /*
    for (auto it = ICUSTAY_ID_TIME_Map.begin(); it != ICUSTAY_ID_TIME_Map.end(); it++)
    {
        printf("id: %d / timestamp : %s\n", (*it).first, (*it).second.c_str());
    }
    */
    
    return true;
}


bool filter(string line){
    string tmp = line;
    for (size_t i = 0; i < 3; i++) tmp = tmp.substr(tmp.find(",") + 1); // for skip before ICUSTAY_ID

    string id_tmp = tmp.substr(0, tmp.find(","));
    if ( id_tmp == "" ) return false; // if ICUSTAY_ID is empty

    int id = stoi(id_tmp);
    if ( ICUSTAY_ID_TIME_Map.find(id) == ICUSTAY_ID_TIME_Map.end() ) return false; // if no ICUSTAY ID

    tmp = tmp.substr(tmp.find(",") + 1); // ITEMID
    tmp = tmp.substr(tmp.find(",") + 1); // CHARTTIME

    cout << "[!] DEBUG " << "\n";
    // convert ICUSTAY TIME to timestamp
    tm ICUSTAY_timestamp;
    //cout << ICUSTAY_ID_TIME_Map[id].c_str() << "\n";
    if (strptime(ICUSTAY_ID_TIME_Map[id].c_str(), "%Y-%m-%d %H:%M:%S", &ICUSTAY_timestamp) == NULL) {
        fprintf(stderr, "ID: %d / 2 strptime error\n", id);
        exit(-1);
    }
    time_t INTIME = mktime(&ICUSTAY_timestamp);

    // convert CHARTEVENT TIME to timestamp
    tm CHART_timestamp;
    //cout << tmp.substr(0, tmp.find(",")).c_str() << "\n";
    if (strptime(tmp.substr(0, tmp.find(",")).c_str(), "%Y-%m-%d %H:%M:%S", &CHART_timestamp) == NULL) {
        fprintf(stderr, "ID: %d / 1 strptime error\n", id);
        exit(-1);
    }
    time_t CHARTTIME = mktime(&CHART_timestamp);

    // make ICUSTAY INCOME TIME + 3h in timestamp
    tm ICUSTAY_timestamp_3h = ICUSTAY_timestamp;
    ICUSTAY_timestamp_3h.tm_hour += 3; // 3 hours
    time_t INTIME_3h = mktime(&ICUSTAY_timestamp_3h);
    
    
    // for debug
    /*
    cout << "CHART :      " << CHARTTIME << "\n";
    cout << "INTIME :     " << INTIME << "\n";
    cout << "INTIME + 3 : " << INTIME_3h << "\n";
    cout << line << "\n";
    */

    // time check
    if ( CHARTTIME < INTIME){
        printf("ID: %d / CHARTIME ERROR", id);
        exit(-1);
    }
    if (INTIME_3h < CHARTTIME) return false;

    return true;
}


int main(int argc, char const *argv[])
{
    if (argc != 4){
        printf("usage : ./chartevents_filter <path-to-raw-chartevents.csv : input> <path-to-filtered-chartevents.csv : output> <path-to-filtering-pair>\n");
        printf("sample: ./chartevents_filter ./dataset/CHARTEVENTS.csv ./filtered_dataset/CHARTEVENTS.csv ./filtered_dataset/ICUSTAY_ID_TIME_PAIR.csv\n");
        return -1;
    }

    infile.open(argv[1]);
    outfile.open(argv[2]);

    signal(SIGINT, sigint_handler);

    printf("[1] INIT FILTER PAIR\n");
    init_filter_pair(argv[3]);


    string tmp;
    int count = 0;

    printf("[2] processing CHARTEVENTS.csv\n");
    getline(infile,tmp); // for first line of csv file
    while (getline(infile, tmp))
    {
        if ( filter(tmp) ){
            outfile << tmp << "\n";
        }

        count += 1;
        if (count % 100000 == 0) {
            printf("\rProgress: %d", count);
            fflush(stdout);
        }
    }
    
    return 0;
}
