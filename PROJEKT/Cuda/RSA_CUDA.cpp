#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>

#define MAX_STR_LEN 10000
#define BLOCKWID 128
// #define BLOCK_SIZE 16
#define GRID_SIZE 8

long int prime(long int);
long int gcd(long int p, long int q);
int publickey(long int p, long int q, long int* exp, long int* mod);
int privatekey(long int p, long int q, long int pubexp, long int* exp, long int* mod);
int encrypt(long int* inmsg, long int, long int, long int* outmsg, size_t len);
int decrypt(long int* inmsg, long int, long int, long int* outmsg, size_t len);
int char2long(char* in, long int* out, bool random_salt=false);
int long2char(long int* in, char* out, bool subtract_pairs=false);

long int fastexp(long int base, long int exp, long int mod);
int BLOCK_SIZE = 16;

int main(int argc, char** argv) {

   long int p,q, pube, pubmod, prive, privmod;
   char inmsg[MAX_STR_LEN];
   long int inmsg_l[MAX_STR_LEN*2];
   char outmsg[MAX_STR_LEN];
   long int outmsg_l[MAX_STR_LEN*2];
   char decrmsg[MAX_STR_LEN];
   long int decrmsg_l[MAX_STR_LEN*2];

   size_t len;

   clock_t encrypt_time;
    BLOCK_SIZE = strtol(argv[1], NULL, 10);

   //myin will take input from a file if specified on the command line and 
   //  from keyboard input (or piped input) if no file is specified
//    std::istream* myin;
//    if (0==argc) myin = &std::cin;
//    else myin = new std::ifstream(argv[1]);
 
   #ifdef __CUDA
   //Wake up the GPU
   long int *dp;
   cudaMalloc(&dp, sizeof(long int));
   cudaFree(dp);
   #endif

   //Get inputs
   // - two prime numbers
   // - a message to be encrypted
//    *myin >> p;
//    if (prime(p)) 
//    {
//       std::cerr << p << " is not prime." << std::endl;
//       return 1;
//    }
//    *myin >> q;
//    if (prime(q)) 
//    {
//       std::cerr << q << " is not prime." << std::endl;
//       return 1;
//    }
    p = 80051;
    q = 3659;
    strcpy(inmsg, "OpenMP (ang. Open Multi-Processing) – wieloplatformowy interfejs programowania aplikacji (API) umożliwiający tworzenie programów komputerowych dla systemów wieloprocesorowych z pamięcią dzieloną. Może być wykorzystywany w językach programowania C, C++ i Fortran na wielu architekturach, m.in. Unix i Microsoft Windows. Składa się ze zbioru dyrektyw kompilatora, bibliotek oraz zmiennych środowiskowych mających wpływ na sposób wykonywania się programu.Dzięki temu, że standard OpenMP został uzgodniony przez głównych producentów sprzętu i oprogramowania komputerowego, charakteryzuje się on przenośnością, skalowalnością, elastycznością i prostotą użycia. Dlatego może być stosowany do tworzenia aplikacji równoległych dla różnych platform, od komputerów klasy PC po superkomputery.OpenMP można stosować do tworzenia aplikacji równoległych działających na wieloprocesorowych węzłach klastrów komputerowych. W tym przypadku stosuje się rozwiązanie hybrydowe, w którym programy są uruchamiane na klastrach komputerowych pod kontrolą alternatywnego interfejsu MPI, natomiast do zrównoleglenia pracy węzłów klastrów wykorzystuje się OpenMP. Alternatywny sposób polegał na zastosowaniu specjalnych rozszerzeń OpenMP dla systemów pozbawionych pamięci współdzielonej (np. Cluster OpenMP[1] Intela).Celem OpenMP jest implementacja wielowątkowości, czyli metody zrównoleglania programów komputerowych, w której główny „wątek programu” (czyli ciąg następujących po sobie instrukcji) „rozgałęzia” się na kilka „wątków potomnych”, które wspólnie wykonują określone zadanie. Wątki pracują współbieżnie i mogą zostać przydzielone przez środowisko uruchomieniowe różnym procesorom. Fragment kodu, który ma być wykonywany równolegle, jest w kodzie źródłowym oznaczany odpowiednią dyrektywą preprocesora. Tuż przed wykonaniem tak zaznaczonego kodu główny wątek rozgałęzia się na określoną liczbę nowych wątków. Każdy wątek posiada unikatowy identyfikator (ID), którego wartość można odczytać funkcją omp_get_thread_num() (w C/C++) lub OMP_GET_THREAD_NUM() (w Fortranie). Identyfikator wątku jest liczbą całkowitą, przy czym identyfikator wątku głównego równy jest 0. Po zakończeniu przetwarzania zrównoleglonego kodu wątki „włączają się” z powrotem do wątku głównego, który samotnie kontynuuje działanie programu i w innym miejscu może ponownie rozdzielić się na nowe wątki.");

//    myin->ignore(INT_MAX,'\n');
//    myin->getline(inmsg, MAX_STR_LEN);
   len = strlen(inmsg);

   //Generate public and private keys from p and q
   publickey(p,q,&pube,&pubmod);
   privatekey(p,q,pube,&prive,&privmod);


   char2long(inmsg, inmsg_l,true);
   
//    clock_t start_clock = clock();
   encrypt(inmsg_l, pube, pubmod, outmsg_l, len*2);
//    encrypt_time = clock() - start_clock;
   
//    std::cout << encrypt_time << std::endl;
}


long int prime(long int p) 
//returns zero for prime numbers
{
   long int j = sqrt(p);
   for (long int z=2;z<j;z++) if (0==p%z) return z;
   return 0;
}

int publickey(long int p, long int q, long int *exp, long int *mod)
//Generates a public key pair
//The modulus is given by (p-1)*(q-1)
//The exponent is any integer coprime of the modulus
{

   *mod = (p-1)*(q-1);
   //Choose an integer near sqrt(mod)
   *exp = (int)sqrt(*mod);
   //Find a coprime near that number 
   while (1!=gcd(*exp,*mod))  
   {
      (*exp)++;
   }
   *mod = p*q;
   return 0;
}

int privatekey(long int p, long int q, long int pubexp, long int *exp, long int *mod)
//Generates a private key pair
//The modulus is given by (p-1)*(q-1)
//The exponent is the number, n, which satisfies (n * pubexp) % mod = 1
{
   *mod = (p-1)*(q-1);
   *exp = 1;
   long int tmp=pubexp;
   while(1!=tmp%*mod)
   {
      tmp+=pubexp;
      tmp%=*mod; //We can exploit the fact that (a*b)%c = ((a%c)*b)%c 
                 //   to keep the numbers from getting too large
      (*exp)++;
   }
   *mod = p*q;
   return 0;
}

#ifndef __CUDA
int encrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
//Encrypt an array of long ints
//exp and mod should be the public key pair
//Each number, c, is encrypted by 
// c' = (c^exp)%mod
{
   #pragma acc parallel loop
   #pragma omp parallel for
   for (int i=0; i < len; i++)
   {
      long int c = in[i];
      #if 0
      out[i] = fastexp(c, exp, mod);
      #else
      //This is the slow way to do exponentiation
      for (int z=1;z<exp;z++)
      {
         c *= in[i];
         c %= mod; //We can exploit the fact that (a*b)%c = ((a%c)*b)%c
                   //   to keep the numbers from getting too large
      }
      out[i] = c; 
      #endif
   }
   out[len]='\0'; //Terminate with a zero
   return 0;
}

int decrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
//Decrypt an array of long ints
//exp and mod should be the private key pair
//Each number, c', is decrypted by 
// c = (c'^exp)%mod
{
   #pragma acc parallel loop
   #pragma omp parallel for
   for (int i=0; i < len; i++)
   {
      long int c = in[i];
      #if 1
      out[i] = fastexp(c, exp, mod);
      #else
      //This is the slow way to do exponentiation
      for (int z=1;z<exp;z++)
      {
         c *= in[i];
         c %= mod; //We can exploit the fact that (a*b)%c = ((a%c)*b)%c
                   //   to keep the numbers from getting too large
      }
      out[i] = c; 
      #endif
   }
   out[len]='\0'; //Terminate with a zero
   return 0;
}
#else //ifndef CUDA
__global__ void decrypt_kernel(long int* inout, long int exp, long int mod, size_t len)
//This is the CUDA "kernel." It will be run on each thread. A CUDA kernel is always
// of type void and specified as __global__
{
   //CUDA threads are divided into "blocks." The block number is stored in blockIdx. 
   //  The thread number within that block is stored in threadIdx. The number of  
   //  threads per block is given by blockDim. The total number of blocks is
   //  gridDim.

   //Here, we assign each thread one number to encrypt/decrypt. Each block gets a 
   //  consecutive set of blockDim.x values. Once each value is encrypted, the
   //  thread jumps ahead by blockDim.x*gridDim.x until all values are encrypted.
   for (int t = threadIdx.x + blockIdx.x*blockDim.x; t<len ;t+=blockDim.x*gridDim.x)
   {
      if (t<len) inout[t] = fastexp(inout[t], exp, mod); 
   } 
}

int encrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
//Encrypt an array of long ints
//exp and mod should be the public key pair
//Each number, c', is decrypted by 
// c = (c'^exp)%mod
{
   long int *d_inout;
   //Allocate memory in the separate memory space of the GPU
   cudaMalloc(&d_inout, sizeof(long int)*len); 

   //copy data to the GPU
   cudaMemcpy(d_inout, in, sizeof(long int)*len, cudaMemcpyHostToDevice); //copy to GPU

    float time;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

   //Launch the kernel on the GPU with 1024 threads arranged in blocks of size BLOCKWID
   decrypt_kernel<<<BLOCK_SIZE, GRID_SIZE>>> (d_inout, exp, mod, len);

    //End timer and put result into time variable
    cudaDeviceSynchronize();			 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    printf("%.4f\n", time);

   //copy data back from GPU
   cudaMemcpy(out, d_inout, sizeof(long int)*len, cudaMemcpyDeviceToHost); //copy from GPU

   out[len]=0; //Terminate with a zero
   cudaFree(d_inout);
   return 0;
}
int decrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
//Decrypt an array of long ints
//exp and mod should be the private key pair
//Each number, c', is decrypted by 
// c = (c'^exp)%mod
{
   long int *d_inout;
   //Allocate memory in the separate memory space of the GPU
   cudaMalloc(&d_inout, sizeof(long int)*len); 

   //copy data to the GPU
   cudaMemcpy(d_inout, in, sizeof(long int)*len, cudaMemcpyHostToDevice); //copy to GPU

   //Launch the kernel on the GPU with 1024 threads arranged in blocks of size BLOCKWID
   decrypt_kernel<<<(1024*128+BLOCKWID-1)/BLOCKWID, BLOCKWID>>> (d_inout, exp, mod, len);

   //copy data back from GPU
   cudaMemcpy(out, d_inout, sizeof(long int)*len, cudaMemcpyDeviceToHost); //copy from GPU

   out[len]=0; //Terminate with a zero
   cudaFree(d_inout);
   return 0;
}
#endif



#ifdef __CUDA
__device__ __host__ 
#endif
long int fastexp(long int base, long int exp, long int mod)
{
   long int out = 1;
   while(exp>0)
   {
      if(1==exp%2) 
      {
         out*=base;
         out%=mod;
      }  
      base=base*base;
      base%=mod;
      exp/=2;
   }
   return out;
}


long int gcd(long int p, long int q)
{
   if (p<q) {long int tmp=p;p=q;q=tmp;}
   while (q!=0)
   {
      long int tmp = q;
      q = p%q;
      p = tmp;
   }

   return p;
}

int long2char(long int* in, char* out, bool subtract_pairs)
{
   while(*in != 0 || *(in+1) != 0) 
   {
      long int r = 0;
      if (subtract_pairs)
      {
         r = *in++;
      }
      *out++ = (char)(*in++)-r;
   }
   *out = '\0';
   return 0;
}

int char2long(char* in, long int* out, bool random_salt)
{
   while(*in != '\0') 
   {
      long int r = 0;
      if (random_salt)
      {
         r = rand()%INT_MAX - INT_MAX/2;
         *out++ = r;
      }
      *out++ = (long int)(*in++) + r;
   }
   *out++ = 0;
   *out = 0;
   return 0;
}
