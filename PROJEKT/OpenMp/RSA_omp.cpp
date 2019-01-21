#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <iostream>
#include <string.h>
#include "omp.h"


#define MAX_STR_LEN 10000

long int prime(long int);
long int gcd(long int p, long int q);
int publickey(long int p, long int q, long int* exp, long int* mod);
int privatekey(long int p, long int q, long int pubexp, long int* exp, long int* mod);
int encrypt(long int* inmsg, long int, long int, long int* outmsg, size_t len);
int decrypt(long int* inmsg, long int, long int, long int* outmsg, size_t len);
int char2long(char* in, long int* out);
int long2char(long int* in, char* out);

int numberOfThreads = 1;

int main(int argc, char** argv) {

   long int p,q, pube, pubmod, prive, privmod;
   char inmsg[MAX_STR_LEN];
   long int inmsg_l[MAX_STR_LEN];
   char outmsg[MAX_STR_LEN];
   long int outmsg_l[MAX_STR_LEN];
   char decrmsg[MAX_STR_LEN];
   long int decrmsg_l[MAX_STR_LEN];

   size_t len;

   numberOfThreads = strtol(argv[1], NULL, 10);

   std::cin >> p;
   if (prime(p)) 
   {
      std::cerr << p << " is not prime." << std::endl;
      return 1;
   }

   std::cin >> q;
   if (prime(q)) 
   {
      std::cerr << q << " is not prime." << std::endl;
      return 1;
   }

   std::cin.ignore(INT_MAX,'\n');
   std::cin.getline(inmsg, MAX_STR_LEN);
   len = strlen(inmsg);

   //Generate public and private keys from p and q
   publickey(p,q,&pube,&pubmod);
   privatekey(p,q,pube,&prive,&privmod);

    //Convert to long ints
   char2long(inmsg, inmsg_l);

   auto startTime = omp_get_wtime();

   encrypt(inmsg_l, pube, pubmod, outmsg_l, len);

   auto endTime = omp_get_wtime();

    std::cout << ((endTime - startTime) * 1000) << std::endl	;
}


long int prime(long int p) 
//returns zero for prime numbers
{
   long int j = sqrt(p);
   for (long int z=2;z<j;z++) if (0==p%z) return z;
   return 0;
}

int publickey(long int p, long int q, long int *exp, long int *mod)
{

   *mod = (p-1)*(q-1);
   *exp = (int)sqrt(*mod);
   while (1!=gcd(*exp,*mod))  
   {
      (*exp)++;
   }
   *mod = p*q;
   return 0;
}

int privatekey(long int p, long int q, long int pubexp, long int *exp, long int *mod)
{
   *mod = (p-1)*(q-1);
   *exp = 1;
   long int tmp=pubexp;
   while(1!=tmp%*mod)
   {
      tmp+=pubexp;
      tmp%=*mod;
      (*exp)++;
   }
   *mod = p*q;
   return 0;
}

int encrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
{
   #pragma omp parallel for schedule(dynamic) num_threads(numberOfThreads)
   for (int i=0; i < len; i++)
   {
      long int c = in[i];
      for (int z=1;z<exp;z++)
      {
         c *= in[i];
         c %= mod; 
      }
      out[i] = c; 
   }
   out[len]='\0'; 
   return 0;
}

int decrypt(long int* in, long int exp, long int mod, long int* out, size_t len)
{
   #pragma omp parallel for
   for (int i=0; i < len; i++)
   {
      long int c = in[i];
      for (int z=1;z<exp;z++)
      {
         c *= in[i];

         c %= mod;
      }
      out[i] = c; 
   }
   out[len]='\0';
   return 0;
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

int long2char(long int* in, char* out)
{
   while(*in != 0) *out++ = (char)(*in++);
   *out = '\0';
   return 0;
}

int char2long(char* in, long int* out)
{
   while(*in != '\0') *out++ = (long int)(*in++);
   *out = 0;
   return 0;
}
