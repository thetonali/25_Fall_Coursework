#include <stdio.h>
#include <unistd.h>

void main(){
    int pid;
    printf("Just one process now.\n");
    printf("Parent calling fork()......\n");
    pid=fork();
    if(pid==0)
        printf("I am the child.\n");
    else if(pid>0)
        printf("I am the parent.\n");
    else
        printf("fork failed.\n");
    printf("Program end.\n");
}