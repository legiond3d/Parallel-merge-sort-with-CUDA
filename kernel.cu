//библиотеки CUDA C
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
//стандартные библиотеки C++
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<clocale>
#include<vector>
#include<Windows.h>
using namespace std;
struct CL
{
	unsigned long long C;
	int L;
	__device__ __host__ CL()
	{
		C = 18446744073709551614;
		L = -1;
	}
	__device__ __host__ CL& operator=(const CL& cl)
	{
		if (this != &cl)
		{
			C = cl.C;
			L = cl.L;
		}
		return *this;
	}
	__device__ __host__ bool operator<=(CL cl)
	{
		return C <= cl.C;
	}
	__device__ __host__ bool operator<(CL cl)
	{
		return C < cl.C;
	}
	__device__ __host__ bool operator>=(CL cl)
	{
		return C >= cl.C;
	}
	__device__ __host__ bool operator>(CL cl)
	{
		return C > cl.C;
	}
	__device__ __host__ bool operator==(CL cl)
	{
		return C == cl.C;
	}
};
//Пирамидальная сортировка
template<class T1>
__host__ __device__ void heap(T1* array, long long n, long long i)
{
	T1 x;
	long long max = i;//наибольший элемент представим корнем 
	long long left = 2 * i + 1;
	long long right = 2 * i + 2;
	if (left < n && array[left] > array[max])
		max = left;
	if (right < n && array[right] > array[max])
		max = right;
	if (max != i)//Если самый большой элемент не корень меняем их местами
	{
		x = array[max];
		array[max] = array[i];
		array[i] = x;
		heap(array, n, max);
	}
}
template<class T1>
__host__ __device__ void heap_sort(T1* array, long long n)
{
	T1 x;
	for (long long i = n / 2 - 1; i >= 0; i--)
		heap(array, n, i);//перегруппировка массива в кучу
	for (long long i = n - 1; i >= 0; i--)
	{
		x = array[i];
		array[i] = array[0];
		array[0] = x;
		heap(array, i, 0);
	}
}
__global__ void merge_sort_sl(CL* cl, CL* sup, int N, int block_size, int chunk, int el)
{
	CL* cl1 = sup;
	CL* cl2 = (CL*)&cl1[N];
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int i, j, k, m;
	int start, mid, end;
	if (tx < N)
	{
		for (int f = tx * el; f < tx * el + el; f++)
			cl1[f] = cl[f];
		__syncthreads();
		for (m = 2 * chunk; m <= block_size; m *= 2)
		{
			//if (tx % m == 0)
			{
				start = tx;
				mid = start + m / 2;
				end = start + m;
				i = start;
				k = start;
				j = mid;
				while (i < mid || j < end)
				{
					if (i < mid && j == end)
						cl2[k] = cl1[i++];
					if (i == mid && j < end)
						cl2[k] = cl1[j++];
					if (i < mid && j < end)
						if (cl1[i] < cl1[j])
							cl2[k] = cl1[i++];
						else
							cl2[k] = cl1[j++];
					k++;
				}
			}
			__syncthreads();
			for (int f = tx * el; f < tx * el + el; f++)
				cl1[f] = cl2[f];
			__syncthreads();
		}
		for (int f = tx * el; f < tx * el + el; f++)
			cl[f] = cl1[f];
		__syncthreads();
	}
}
__global__ void MergeSort(CL* cl, int N, int chunk)
{
	extern __shared__ CL cl1[];
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int start, mid, end, k, j, i;
	start = tx * chunk;
	if (start < N)
	{
		for (int f = start; f < start + chunk; f++)
			cl1[f] = cl[f];
		__syncthreads();
		mid = min(start + chunk / 2, N);
		end = min(start + chunk, N);
		k = i = start;
		j = mid;
		while (i < mid || j < end)
		{
			if (j == end)
				cl1[k] = cl[i++];
			else if (i == mid)
				cl1[k] = cl[j++];
			else if (cl[i] < cl[j])
				cl1[k] = cl[i++];
			else
				cl1[k] = cl[j++];
			k++;
		}
		__syncthreads();
		for (int f = start; f < start + chunk; f++)
			cl[f] = cl1[f];
	}
}
__global__ void merge_sort(CL* cl, int N, int block_size, int chunk, int el)//сортировка слиянием в рамках блоков потоков (worked with shared memory)
{
	extern __shared__ CL cache[];//динамически выделенная разделяемая память для 2 массивов
	CL* cl1 = cache;
	CL* cl2 = (CL*)& cl1[block_size];
	int tx = threadIdx.x;
	int ktx = tx + blockDim.x * blockIdx.x;
	int i, j, k, m;
	int start, mid, end;
	if (ktx < N)
	{
		if (el > 1)
			for (int f = tx * el; f < tx * el + el; f++)
				cl1[f] = cl[f + blockDim.x * blockIdx.x * el];
		else
			cl1[tx] = cl[ktx];
		__syncthreads();
		for (m = 2 * chunk; m <= block_size; m *= 2)
		{
			if (tx % m == 0)
			{
				start = tx;
				mid = start + m / 2;
				end = start + m;
				i = start;
				k = start;
				j = mid;
				while (i < mid || j < end)
				{
					if (i < mid && j == end)
						cl2[k] = cl1[i++];
					else if (i == mid && j < end)
						cl2[k] = cl1[j++];
					else if (i < mid && j < end)
						if (cl1[i] < cl1[j])
							cl2[k] = cl1[i++];
						else
							cl2[k] = cl1[j++];
					k++;
				}
			}
			__syncthreads();
			if (el > 1)
				for (int f = tx * el; f < tx * el + el; f++)
					cl1[f] = cl2[f];
			else
				cl1[tx] = cl2[tx];
			__syncthreads();
		}
		if (el > 1)
			for (int f = tx * el; f < tx * el + el; f++)
				cl[f + blockDim.x * blockIdx.x * el] = cl1[f];
		else
			cl[ktx] = cl1[tx];
		__syncthreads();
	}
}
__host__ __device__ void merge(CL* cl, CL* support, int start, int mid, int end)
{
	int ti = start, i = start, j = mid;
	while (i < mid || j < end)
	{
		if (j == end) support[ti] = cl[i++];
		else if (i == mid) support[ti] = cl[j++];
		else if (cl[i] < cl[j]) support[ti] = cl[i++];
		else support[ti] = cl[j++];
		ti++;
	}
	for (ti = start; ti < end; ti++)
		cl[ti] = support[ti];
}
__host__ __device__ void merge_block(CL* cl, CL* support,int N, int BLOCK_SIZE, int GRID_SIZE)//слияние отсортированных блоков
{
	for (int i = 0; i * GRID_SIZE <= N; i++)
	{
		int start = i * GRID_SIZE, end, mid;
		if (start >= N) return;
		mid = min(start + GRID_SIZE / 2, N);
		end = min(start + GRID_SIZE, N);
		merge(cl, support, start, mid, end);
	}
}
__global__ void parallel_merge_sort(CL* cl, CL* sup, int N, int chunk)//альтернатива, без использования разделяемой памяти
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int start = tx * chunk;
	if (start < N)
	{
		int mid = min(start + chunk / 2, N);
		int end = min(start + chunk, N);
		int k = start;
		int i = start;
		int j = mid;
		while (i < mid || j < end)
		{
			if (j == end)
				sup[k] = cl[i++];
			else if (i == mid)
				sup[k] = cl[j++];
			else if (cl[i] < cl[j])
				sup[k] = cl[i++];
			else
				sup[k] = cl[j++];
			k++;
		}
		for (int f = start; f < start + chunk; f++)
			cl[f] = sup[f];
	}
}
__host__ void test(CL* cl, CL* sup,  int N, int threads, int blocks)
{
	for (int chunk = 2; chunk <= N; chunk *= 2, blocks /= 2)
	{
		parallel_merge_sort << < blocks, threads >> > (cl, sup, N, chunk);
	}
}
int main()
{
	setlocale(LC_ALL, "Rus");
	int ndevice;
	cudaGetDeviceCount(&ndevice);
	for (int i = 0; i < ndevice; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device number: %d\n", i);
		printf("Device name: %s\n", prop.name);
	}
	int N = 1024 * 1024;
	cout << "N = " << N << endl;
	int BLOCK_SIZE = 1024;
	int GRID_SIZE = N % BLOCK_SIZE == 0 ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1;
	cout << "BLOCK_SIZE = " << BLOCK_SIZE << ", GRID_SIZE = " << GRID_SIZE << ", GRID_SIZE * BLOCK_SIZE = " << GRID_SIZE * BLOCK_SIZE << endl;
	//for (BLOCK_SIZE; GRID_SIZE > 1024; BLOCK_SIZE *= 2)
	//	GRID_SIZE = (N + BLOCK_SIZE) / BLOCK_SIZE;
	CL* cl = new CL[N];
	CL* support = new CL[N];
	CL* cl2 = new CL[N];
	srand(time(NULL));
	ofstream supm("support.txt");
	ofstream clm("clm.txt");
	cout << "Original : " << endl;
	for (int i = 0; i < N; i++)
	{
		cl[i].C = 0 + rand() % 10000;
		support[i].C = cl[i].C;
		cl2[i].C = cl[i].C;
		cl[i].L = i;
		support[i].L = cl[i].L;
		cl2[i].L = cl[i].L;
		supm << support[i].C << " | " << support[i].L << endl;
		clm << cl[i].C << " | " << cl[i].L << endl;
		//cout << cl[i].C << "|" << cl[i].L;
		//if (i == N - 1)
		//	cout << endl;
		//else
		//	cout << ", ";
	}
	supm.close();
	clm.close();

	//пирамидальная сортировка на процессоре
	clock_t start1;
	start1 = clock();
	heap_sort(support, N);
	cout << "Для последовательной сортировки с помощью кучи потребовалось: t = " << setprecision(15) << (double)(clock() - start1) / 1000 << "c." << endl;

	CL* device;
	CL* sup;
	CL* device2;
	CL* sup2;
	cudaMalloc((void**)&device, N * sizeof(CL));
	cudaMalloc((void**)&sup, N * sizeof(CL));
	cudaMalloc((void**)&device2, N * sizeof(CL));
	cudaMalloc((void**)&sup2, N * sizeof(CL));
	cudaMemcpy(device, cl, N * sizeof(CL), cudaMemcpyHostToDevice);
	cudaMemcpy(device2, cl2, N * sizeof(CL), cudaMemcpyHostToDevice);
	
	clock_t start;
	start = clock();
	int grid = GRID_SIZE;//количество блоков, каждый из которых будет сортироваться независимо
	int BLOCK = BLOCK_SIZE;//количество потоков в каждом блоке
	int block = BLOCK_SIZE;//количество элементов которые требуется отсортировать на данной итерации
	int chunk = 1;//показывает сколько элементов было отсортировано на предыдущей итерации
	int el = 1;//количество элементов для обработки одним потоком
	while (grid >= 1)
	{
		if (2 * block * sizeof(CL) <= 96 * 1024)//если есть возможность уместить массивы в разделяемой памяти
		{
			cudaFuncSetAttribute(merge_sort, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * block * sizeof(CL));//позволяет использовать до 96КБ на блок на архитектуре Volta и выше
			merge_sort << < grid, BLOCK, 2 * block * sizeof(CL) >> > (device, N, block, chunk, el);
		}
		else
		{
			//merge_sort_sl << < grid, BLOCK >> > (device, sup, N, block, chunk, el);
			parallel_merge_sort << < grid, BLOCK >> > (device, sup, N, block);
		}
		//cout << "Сколько разделяемой памяти запросили на итерации shared memory = " << sizeof(CL) * 2 * block / (1024) << endl;
		//cout << "Какая итерация = " << el << endl;
		grid /= 2;//уменьшаем в 2 раза количество блоков (объединяем блоки попарно)
		el *= 2;
		chunk = block;
		block *= 2;
	}
	cout << "Для параллельной сортировки слиянием потребовалось: t = " << setprecision(15) << (double)(clock() - start) / 1000 << "c." << endl;
	cudaMemcpy(cl, device, N * sizeof(CL), cudaMemcpyDeviceToHost);

	clock_t start2;
	start2 = clock();
	grid = GRID_SIZE % 2 == 0 ? GRID_SIZE / 2 : GRID_SIZE / 2 + 1;
	block = BLOCK_SIZE;
	chunk = 2;
	for (chunk = 2; chunk <= N; chunk *= 2)
	{
		grid = N / (chunk * block); 
		parallel_merge_sort << < grid, block >> > (device2, sup2, N, chunk);
	}
	/*while (grid >= 1)
	{
		MergeSort << < grid, block, 2 * block * chunk * sizeof(CL) >> > (device2, N, chunk);
		grid = grid % 2 == 0 ? grid / 2 : grid / 2 + 1;
		chunk = N / (grid * block);
	}*/
	cout << "Параллельная сортировка без shared memory: t = " << setprecision(15) << (double)(clock() - start2) / 1000 << "c." << endl;
	cudaMemcpy(cl2, device2, N * sizeof(CL), cudaMemcpyDeviceToHost);

	double error = 0.0;
	for (int i = 0; i < N; i++)
	{
		error += support[i].C - cl[i].C;
	}
	cout << "error 1 = " << error << endl;
	error = 0.0;
	for (int i = 0; i < N; i++)
		error += support[i].C - cl2[i].C;
	cout << "error 2 = " << error << endl;

	/*cout << "Sorted cl: " << endl;
	for (int i = 0; i < N; i++)
	{
		cout << cl[i].C << "|" << cl[i].L;
		if (i == N - 1)
			cout << endl;
		else
			cout << ", ";
	}
	cout << "Sorted sup (pyramidal)):" << endl;
	for (int i = 0; i < N; i++)
	{
		cout << support[i].C << "|" << support[i].L;
		if (i == N - 1)
			cout << endl;
		else
			cout << ", ";
	}*/
	ofstream pyram("pyram.txt");
	ofstream mergem("mergem.txt");
	for (int i = 0; i < N; i++)
	{
		pyram << i << ") " << support[i].C << " | " << support[i].L << endl;
		mergem << i << ") " << cl[i].C << " | " << cl[i].L << endl;
	}
	pyram.close();
	mergem.close();
	delete[] cl;
	delete[] support;
	delete[] cl2;
	cudaFree(device);
	cudaFree(sup);
	cudaFree(device2);
	cudaFree(sup2);
	system("pause");
	return 0;
}