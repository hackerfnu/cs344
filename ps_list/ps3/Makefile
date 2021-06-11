HIPCC=hipcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/home/anush/stanley/parallel_compute_course/opencv-install/lib
OPENCV_INCLUDEPATH=/home/anush/stanley/parallel_compute_course/opencv-install/include/opencv4

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

HIP_INCLUDEPATH=/opt/rocm/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

HIPCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -m64 --std=c++11

GCC_OPTS=-O3 -Wall -Wextra -m64

DEFS=-D__HIP_PLATFORM_HCC__

student: main.o student_func.o HW3.o loadSaveImage.o compare.o reference_calc.o Makefile
	$(HIPCC) -o HW3 main.o student_func.o HW3.o loadSaveImage.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(HIPCC_OPTS) $(DEFS)

main.o: main.cpp timer.h utils.h reference_calc.h compare.h
	g++ -c main.cpp $(GCC_OPTS) -I $(HIP_INCLUDEPATH) $(DEFS)

HW3.o: HW3.cpp loadSaveImage.h utils.h
	$(HIPCC) -c HW3.cpp -I $(OPENCV_INCLUDEPATH) $(HIPCC_OPTS) $(DEFS)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(HIP_INCLUDEPATH) $(DEFS)

compare.o: compare.cpp compare.h
	g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(HIP_INCLUDEPATH) $(DEFS)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(HIP_INCLUDEPATH) $(DEFS)

student_func.o: student_func.cpp utils.h
	$(HIPCC) -c student_func.cpp $(HIPCC_OPTS) $(DEFS)

clean:
	rm -f *.o hw
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
