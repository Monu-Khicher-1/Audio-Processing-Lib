## Audio Processing Library
Using basic functions of Subtask1 and Subtask2 We had formed a audio processing library.
### Files 
*Files:*
- audio.cpp : Contains function audiolibAPI.
- audio.h   : function for externing audiolibAPI.
- main.cpp  : File which will use audio processing library.
- weight_dnn.h : Contains weight and bias matrixes.

### How to run Codes
- Open your folder and run commands as below:
~~~
$export LIBROOT=(path to the dirctory where all files are stored.)
$make
$./test audiosamplefile outputfile
~~~
- This will generate different files in directory `audio.o` and `libaudio.so`
- Output to the console will be of type `audiosamplefile yes unknown left 0.999690 0.000288 2.21e-5`.
- And a outputfile will be generated if it doesn't exist. If it is already present then output will be appended to it.
- `audiosamplefile` is text file which contains audio features.
- Also 2 extra files will be produced "probability.txt" and "maxprobability.txt". Which are only for analysis purpose.
