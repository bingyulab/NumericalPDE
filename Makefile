
.PHONY: all clean build

all: clean build

clean:
	@echo "Cleaning previous build..."
	rm -rf cpp/build

build:
	@echo "Building the project..."
	mkdir -p cpp/build
	cd cpp/build && cmake .. && make