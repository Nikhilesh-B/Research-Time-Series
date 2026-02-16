# Hamilton GNP Data Generating Process (C++)
# Usage: make && ./hamilton_gnp_dgp [output.csv] [seed]

CXX = g++
CXXFLAGS = -O3 -Wall

hamilton_gnp_dgp: hamilton_gnp_dgp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f hamilton_gnp_dgp

run: hamilton_gnp_dgp
	./hamilton_gnp_dgp ./gnp_data_created_cpp.csv 42

.PHONY: clean run
