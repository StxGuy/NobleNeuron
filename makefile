TARGET = neural
F90_SOURCE =  ./source/activ.f90 \
              ./source/loss.f90 \
              ./source/layer.f90 \
              ./source/neural.f90 \
              ./source/XOR.f90 \
              
# Change last file accordingly: XOR.f90 or autoencoder.f90              
              

OBJ = $(subst .f90,.o,$(subst source,objects,$(F90_SOURCE)))
FC = gfortran
FC_FLAGS = -O3 \

MOD = mods           
           
RM = rm -rf           

all : display $(TARGET)
	

display:
	@clear
	@mkdir -p objects
	@mkdir -p mods
	
	@echo ".------------------------------------."
	@echo "| Compiling: NobleNeuron             |#"
	@echo "| ---------                          |#"
	@echo "|                                    |#"
	@echo "| By: Prof. Carlo R. da Cunha        |#"
	@echo "|                                    |#"
	@echo "| Created: Mar/2020                  |#"
	@echo "| Revision: Nov/2021                 |#"
	@echo "'------------------------------------'#"
	@echo "  #####################################"
	@echo ""


$(TARGET): $(OBJ)
	@echo "# Linking $@..."
	$(FC) $^ $(FC_FLAGS) -o $@ -I $(MOD)
	@echo ""

./objects/%.o: ./source/%.f90
	@echo "# Building target: $<"
	$(FC) $< $(FC_FLAGS) -c -o $@ -J $(MOD) 
	@echo ""
		
	
clean:
	@$(RM) ./objects/*.o $(TARGET) *~
	@$(RM) ./mods/*.mod $(TARGET) *~
	@rmdir objects
	@rmdir mods
	
.PHONY: all clean	
	
