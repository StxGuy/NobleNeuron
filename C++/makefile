TARGET = NobleNeuron

# Source files
CPP_SOURCE =  ./source/activ.cpp \
	      ./source/loss.cpp \
	      ./source/ff_layer.cpp \
	      ./source/conv_layer.cpp \
	      ./source/pooling_layer.cpp \
	      ./source/FeedForward.cpp \
	      ./source/Convolutional.cpp \

# Include files
INCLUDE = ./include \

              
OBJ = $(subst .cpp,.o,$(subst source,objects,$(CPP_SOURCE)))
CC = g++

# Compiler flags:
#   -g...: debugging
#   -Wall: enable all warning messages
#   -O2: speed!
CPP_FLAGS = -g -Wall -std=c++11 -O2

# Libraries
LIBRARIES = -larmadillo

DIR = '/usr/local/include/NobleNeuron_headers/'
           
RM = rm -rf           

all : display $(TARGET)
	

display:
	@clear
	@mkdir -p objects
		
	@echo ".------------------------------------."
	@echo "| Compiling: NobleNeuron++           |#"
	@echo "| ---------                          |#"
	@echo "|                                    |#"
	@echo "| By: Prof. Carlo R. da Cunha, Ph.D. |#"
	@echo "|                                    |#"
	@echo "| Created: Jan/2022                  |#"
	@echo "| Revision: Jan/2022                 |#"
	@echo "'------------------------------------'#"
	@echo "  #####################################"
	@echo ""


$(TARGET): $(OBJ)
	

./objects/%.o: ./source/%.cpp
	@echo "# Building target: $<"
	$(CC) $< $(CPP_FLAGS) -I$(INCLUDE) -c -o $@ $(LIBRARIES)
	@echo ""
		
install: 
	@clear
	@echo "Creating static library..."
	@ar rcs lib$(TARGET).a objects/*.o
		
ifeq ("$(wildcard $(DIR))","")
	@echo "Directory already exists."
else
	@echo "Creating directory in /usr/local/include/NobleNeuron_headers..."
	sudo mkdir /usr/local/include/NobleNeuron_headers
endif
	
	@echo "# Copying headers ..."
	@sudo cp include/NobleNeuron.hpp /usr/local/include/NobleNeuron
	@sudo cp include/NobleNeuron_headers/* /usr/local/include/NobleNeuron_headers
	@echo "# Copying binary..."
	@sudo mv libNobleNeuron.a /usr/local/lib
		
clean:
	@$(RM) ./objects/*.o $(TARGET) *~
	@rmdir objects
	@rm *.a
	
.PHONY: all clean	
	
