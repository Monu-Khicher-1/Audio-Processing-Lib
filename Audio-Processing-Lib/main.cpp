// This file contains codes for DNN which will take audio file as input and outputs the probability of 12 keywords.

//==============================================================================================================================
//
//==============================================================================================================================
//

#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include "/usr/include/mkl/mkl.h"
#include "audio.h"

using namespace std;

int main(int argc,char* argv[]){
    string st1="";
    string st2="";
    const char* featurefile;
    
    if(argc==3)
    {
        featurefile=argv[1];
        st2=string(argv[2]);
    }
    else{
        return 0;
    }

    // pred_t* pred;
    pred_t* p;
    libaudioAPI(featurefile,p);
    cout<<".........======........"<<endl;
    int index1=p[0].label;
    int index2=p[1].label;
    int index3=p[2].label;
    
    float p1=p[0].prob;
    float p2=p[1].prob;
    float p3=p[2].prob;
    cout<<".........======........"<<endl;
    string ps1,ps2,ps3;
    ps1=p1;
    ps2=p2;
    ps3=p3;
    cout<<".........======........"<<endl;
    st1=featurefile;
    string arr[12] = { "silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go" };
    string result=st1 + " " + arr[index1] +" " + arr[index2]+ " " + arr[index3]+" " + ps1+" "+ps2+" "+ps3;

    cout<<result<<endl;
    
    ifstream myfile(st2);
    if(!myfile.is_open()){
        // make a new file
        ofstream myfile1;
        myfile1.open(st2);
        myfile1<<result<<"\n";
        myfile1.close();
        return 0;
    }
    else{
        myfile.close();
        ofstream fileOUT("filename.txt", ios::app);
        fileOUT<<result<<"\n";
        fileOUT.close();
        return 0;
    }
}


