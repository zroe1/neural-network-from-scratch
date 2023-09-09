all: linalg grads outputs train

linalg: linalg.c tests/linalg_tests.c network_funcs.c 
	clang -o linalg linalg.c tests/linalg_tests.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

outputs: linalg.c tests/network_outputs.c network_funcs.c 
	clang -o outputs linalg.c tests/network_outputs.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

grads: linalg.c tests/grads_tests.c network_funcs.c 
	clang -o grads linalg.c tests/grads_tests.c network_funcs.c -L/opt/homebrew/lib -lcriterion -I/opt/homebrew/include

train: train.c linalg.c network_funcs.c
	clang -o train train.c linalg.c network_funcs.c

clean:
	rm -f linalg grads outputs train

