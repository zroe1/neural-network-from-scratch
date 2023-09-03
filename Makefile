all: linalg grads

linalg: linalg.c tests/linalg_tests.c network_funcs.c 
	clang -o linalg linalg.c tests/linalg_tests.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

grads: linalg.c tests/grads_tests.c network_funcs.c 
	clang -o grads linalg.c tests/grads_tests.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

clean:
	rm -f *.o linalg grads

