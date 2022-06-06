#include <iostream>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include "/usr/include/mkl/mkl.h"
#include "weight_dnn.h"

using namespace std;

typedef struct{
    int label;
    float prob;
}pred_t;



// ===========================================================================
//                FULLYCONNECTED AND RELU AND SOFTMAX
//============================================================================


//==============
//FULLYCONNECTED
//==============
//


int fullyconnected(string inputmatrix,float weightmatrix[],float biasmatrix[],string outputmatrix,int m, int n ,int k)
{
 
    //  Codes for mkl 
    double *A,*B,*C;
    double alpha,beta;

    alpha=1.0;
    beta=1.0;

    A=(double *)mkl_malloc(m*k*sizeof(double),64);
    B=(double *)mkl_malloc(k*n*sizeof(double),64);
    C=(double *)mkl_malloc(m*n*sizeof(double),64);
    
    if (A==NULL || B==NULL || C==NULL){
        // mkl_free(A);
        // mkl_free(B);
        // mkl_free(C);
        return 0;
    }
//----------------------------------------------------------------
// A=>

    string line;
    ifstream myfile(inputmatrix);
    if(!myfile.is_open())
    {
        cout<<"File can't be opened!"<<inputmatrix<<endl;
        return 0;
    }
    getline(myfile,line);
    stringstream ss(line);
    int i=0;
    int j=0;
    while (ss.good() && j<m*k) {
        string substr;
        getline(ss, substr, ' ');
        float a=std::stof(substr);
        A[j]=a;
        j++;       
    }
    myfile.close();

//------------------------------------------

   for (int i=0;i<k*n;i++){
       B[i]=weightmatrix[i];
   }
//-------------------------------------------
    for (int i=0;i<m*n;i++){
       C[i]=biasmatrix[i];
    }
//-----------------------------------------------------------------
//

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);

    ofstream myfile3;
    myfile3.open(outputmatrix);
    i=0;
    while(i<m*n)
    {
        float elem=C[i];  
        myfile3<<elem<<" ";
        i++;
    }
    //mkl_free(A);
    // mkl_free(B);
    // mkl_free(C);

    myfile3.close();   
    return 0;
}

//
//=================
//  Relu
//=================


int Relu(string inputfile,string outputfile,int n){
    string line;
    ifstream myfile(inputfile);
    if(!myfile.is_open())
    {
        cout<<"File can't be opened!"<<endl;
        return 0;
    }
    getline(myfile,line);
    stringstream ss(line);
    int i=0;
    float mat[n];
    while (ss.good() && i<n) {
        string substr;
        getline(ss, substr, ' ');
        float a=std::stof(substr);
        mat[i]=a;
        i++;
    }
    myfile.close();

    ofstream myfile3;
    myfile3.open(outputfile);
    i=0;
    while(i<n)
    {
        float elem=0;
        if(mat[i]>0){
            elem=mat[i];  
        }
        myfile3<<elem<<" ";
        i++;
    }
    myfile3.close();
    return 0;

}


vector<float>  softmax(float a[],int n)
{
    vector<float> v;
    float s=0;
    for(int i=0;i<n;i++)
    {
        float x=a[i];
        s+=exp(x);
    }
    for(int i=0;i<n;i++)
    {
        float x=a[i];
        float y= exp(x)/s;
        v.push_back(y);
    }
    return v;
}

//===============
// Probability
//===============
//


int probability(vector<float>(*func)(float[],int),string inputvector,string outputvector,int n)
{
    string line;
    ifstream myfile(inputvector);
    if(!myfile.is_open())
    {
        cout<<"File can't be opened!"<<endl;
        return 0;
    }
    getline(myfile,line);
    float mat1[n];
    stringstream ss(line);
    int j=0;
    while (ss.good() && j<n) {
        string substr;
        getline(ss, substr, ' ');
        float a=std::stof(substr);
        mat1[j]=a;
        j++;
    }
    myfile.close();

    vector<float> v=func(mat1,n);

    ofstream myfile1;
    myfile1.open(outputvector);

    for(float num:v)
    {
        myfile1<<num<<" ";
    }
    myfile1.close();
    return 0;
}


int maxfn(string inputfile,string outfile,int n){
    string line;
    ifstream myfile(inputfile);
    if(!myfile.is_open())
    {
        cout<<"File can't be opened!"<<inputfile<<endl;
        return 0;
    }
    float arr[n];
    getline(myfile,line);
    stringstream ss(line);
    int j=0;
    while (ss.good() && j<n) {
        string substr;
        getline(ss, substr, ' ');
        float a=std::stof(substr);
        arr[j]=a;
        j++;
    }
    myfile.close();
    
    float max1 =0;
    int maxInd1=0;
    float max2 =0;
    int maxInd2=0;
    float max3 =0;
    int maxInd3=0;


    for(int i=0;i<n;i++){
        if(max1 <arr[i]){
            max1 = arr[i];
            maxInd1= i;
        }
    }

    for(int i=0;i<n;i++){
        if(max2 <arr[i] && arr[i] < max1){
            max2 = arr[i];
            maxInd2= i;
        }
    }

    for(int i=0;i<n;i++){
        if(max3 <arr[i] && arr[i] < max2){
            max3 = arr[i];
            maxInd3= i;
        }
    }
    
    ofstream myfile1;
    myfile1.open(outfile);

    
    myfile1<<maxInd1<<"\n";
    myfile1<<maxInd2<<"\n";
    myfile1<<maxInd3<<"\n";
    myfile1<<max1<<"\n";
    myfile1<<max2<<"\n";
    myfile1<<max3<<"\n";
    
    myfile1.close();
    return 0;


}

//===================================================================================================================================
//                                         libaudioAPI
//===================================================================================================================================


pred_t* libaudioAPI(const char* audiofeature, pred_t *pred){
    string audiofile=audiofeature;

    float weight1[]=IP1_WT;
    float weight2[]=IP2_WT;
    float weight3[]=IP3_WT;
    float weight4[]=IP4_WT;

    float bias1[]=IP1_BIAS;
    float bias2[]=IP2_BIAS;
    float bias3[]=IP3_BIAS;
    float bias4[]=IP4_BIAS;
   

    fullyconnected(audiofile,weight1,bias1,"output1.txt",1,144,250);
    Relu("output1.txt","inp2.txt",144);
    fullyconnected("inp2.txt",weight2,bias2,"output2.txt",1,144,144);
    Relu("output2.txt","inp3.txt",144);
    fullyconnected("inp3.txt",weight3,bias3,"output3.txt",1,144,144);
    Relu("output3.txt","inp4.txt",144);
    fullyconnected("inp4.txt",weight4,bias4,"output4.txt",1,12,144);
    probability(softmax,"output4.txt","probability.txt",12);
    maxfn("probability.txt","maxprobablity.txt",12);
    

    string line;
    ifstream myfile("maxprobablity.txt");
    if(!myfile.is_open())
    {
        cout<<"File can't be opened....!"<<endl;
        return 0;
    }
    int a;
    float b;

    
    getline(myfile,line);
    a=std::stoi(line);
    pred[0].label=a;
    
    getline(myfile,line);
    a=std::stoi(line);
    pred[1].label=a;

    getline(myfile,line);
    a=std::stoi(line);
    pred[2].label=a;

    getline(myfile,line);
    b=std::stof(line);
    pred[0].prob=b;
    
    getline(myfile,line);
    b=std::stof(line);
    pred[1].prob=b;

    getline(myfile,line);
    b=std::stof(line);
    pred[2].prob=b;
    myfile.close();

    // Clearing all extra files 
    char filename[] = "inp2.txt";
    remove(filename);
    char filename1[] = "inp3.txt";
    remove(filename1);
    char filename2[] = "inp4.txt";
    remove(filename2);
    char filename3[] = "output1.txt";
    remove(filename3);
    char filename4[] = "output2.txt";
    remove(filename4);
    char filename5[] = "output3.txt";
    remove(filename5);
    char filename6[] = "output4.txt";
    remove(filename6);
    //char filename7[] = "maxprobablity.txt";
    //remove(filename7);
    // char filename8[] = "probability.txt";
    // remove(filename8); 
   
    return pred;

}


