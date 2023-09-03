all: test

test: linalg.c tests/linalg_tests.c network_funcs.c 
	clang -o test linalg.c tests/linalg_tests.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

clean:
	rm -f *.o test

