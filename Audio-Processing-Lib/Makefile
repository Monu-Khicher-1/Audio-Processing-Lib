MKLROOT=/usr/include/mkl
MKLlink=-I"${MKLROOT}" -L${MKLROOT}/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
all:
	g++ -c -Wall -Werror -fpic audio.cpp ${MKLlink}
	g++ -shared -o libaudio.so audio.o
	g++ -L${LIBROOT} -Wl,-rpath=${LIBROOT} -Wall -o test main.cpp -laudio ${MKLlink}

	