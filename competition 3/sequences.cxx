// Sample code to perform I/O:
#include <stdio.h>

typedef int itype;

int *a,*b,*c;

int n,m;
int k;

long div_count = 0;

int f(int i, int j) {
	return a[i-1] * j *j + b[i-1];
}

int is_seq_divisible(itype* is, itype* js) {
	int total = 0;
	int i = 0;
	for (i=0;i<m;i++) {
		int this_val = f(is[i], js[i]);
		total += this_val;
	}
	if (total % k == 0) {
		return 1;
	}
	return 0;
}

int print_sequence(itype* is, itype* js, int len) {
	int i;
	for (i=0;i<len;i++) {
		printf("(%d,%d)",is[i], js[i]);
	}
	printf("\n");
}


int iterate_sequence(int pair_num, int total_to_here) {
	int i =0;
	int j =0;
	int this_val, cur_total;
	for (i=1;i<=n;i++) {
		for (j=1;j<=c[i-1];j++) {
			this_val = f(i,j);
			cur_total = total_to_here + this_val;
			if (pair_num == (m-1)) {
				// last pair, check
				if (cur_total %k == 0) {
					div_count += 1;
				}
			} else {
				iterate_sequence(pair_num+1, cur_total);
			}
		}	
	}	
}


int main(){
	scanf("%d %d %d", &n, &m, &k);              			// Reading input from STDIN
//	printf("n: %d\n",n);
//	printf("m: %d\n",m);
//	printf("k: %d\n",k);

	a = new int[n];
	b = new int[n];
	c = new int[n];

	int i = 0;

	for (i=0;i<n;i++) {
		int thisa, thisb, thisc;
		scanf("%d %d %d", &thisa, &thisb, &thisc);              			// Reading input from STDIN
		a[i] = thisa;
		b[i] = thisb;
		c[i] = thisc;
	}

	// data loaded
	// start generating all sequences of length m
	// these hold the sequence values

	// range for i is 1->n
	// range for j is 1->c[i]
	// generate all sequences
	// check if each sequence is divisible by k
	// increment count

//	int b = is_seq_divisible(is, js);
//	printf("%d\n",b);
	iterate_sequence(0, 0);
	int small_div_count = div_count % 1000000007;
	printf("%d\n",small_div_count);
//	printf("%d\n",sizeof(long));

//	printf("f(1,1)=%d\n",f(1,1));
//	printf("f(1,2)=%d\n",f(1,2));
}
