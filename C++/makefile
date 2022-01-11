TARGET = neural

# Source files
CPP_SOURCE =  ./source/activ.cpp \
	      ./source/layer.cpp \
              ./source/neural.cpp \
              ./source/loss.cpp \
              ./source/main.cpp \

# Include files
INCLUDE = ./include \

              
OBJ = $(subst .cpp,.o,$(subst source,objects,$(CPP_SOURCE)))
CC = g++

# Compiler flags:
#   -g...: debugging
#   -Wall: enable all warning messages
CPP_FLAGS = -g -Wall -std=c++11 -O2

# Libraries
LIBRARIES = -larmadillo

           
RM = rm -rf           

all : display $(TARGET)
	

display:
	@clear
	@mkdir -p objects
		
	@echo ".------------------------------------."
	@echo "| Compiling: NobleNeuron++           |#"
	@echo "| ---------                          |#"
	@echo "|                                    |#"
	@echo "| By: Prof. Carlo R. da Cunha        |#"
	@echo "|                                    |#"
	@echo "| Created: Jan/2022                  |#"
	@echo "| Revision: Jan/2022                 |#"
	@echo "'------------------------------------'#"
	@echo "  #####################################"
	@echo ""


$(TARGET): $(OBJ)
	@echo "# Linking $@..."
	$(CC) $^ $(CPP_FLAGS) -I$(INCLUDE) -o $@ $(LIBRARIES)
	@echo ""

./objects/%.o: ./source/%.cpp
	@echo "# Building target: $<"
	$(CC) $< $(CPP_FLAGS) -I$(INCLUDE) -c -o $@ 
	@echo ""
		
	
clean:
	@$(RM) ./objects/*.o $(TARGET) *~
	@$(RM) ./mods/*.mod $(TARGET) *~
	@rmdir objects
	@rmdir mods
	
.PHONY: all clean	
	
